// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

#[cfg(windows)]
use std::path::PathBuf;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

// Backend (uvicorn) sürecini başlatıp Windows Job Object'e bağlayan modül.
// Amaç: uygulama nasıl ölürse ölsün (normal kapanma, panik, taskkill, çökme)
// backend süreci de ölsün ki modeller VRAM/RAM/CPU'da asılı kalmasın.
// Süreç ölünce OS bu kaynakları zaten otomatik geri alır; bizim tek işimiz
// sürecin ölmesini garanti etmek.
#[cfg(windows)]
mod backend_proc {
    use std::ffi::c_void;
    use std::fs::File;
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};

    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    // Konsol penceresi açma (CreateProcess bayrağı). Logları dosyaya
    // yönlendiriyoruz, ekstra konsol penceresine gerek yok.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    // Job handle'ını uygulama ömrü boyunca canlı tutan sarmalayıcı.
    // Tauri state'inde tutulur; uygulama kapanınca Drop çalışır, handle
    // kapanır → job kapanır → KILL_ON_JOB_CLOSE backend'i öldürür.
    // Çökme/zorla kapatmada da OS handle'ı kapatır, sonuç aynı.
    // Child'ı da içinde tutuyoruz ki süreç tanıtıcısı erken düşmesin.
    pub struct BackendGuard {
        job: HANDLE,
        _child: Child,
    }

    // HANDLE ham bir pointer; Tauri managed state Send+Sync ister.
    // Tek bir guard var ve paylaşımlı mutasyon yok, bu yüzden güvenli.
    unsafe impl Send for BackendGuard {}
    unsafe impl Sync for BackendGuard {}

    impl Drop for BackendGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.job);
            }
        }
    }

    pub fn spawn(project_root: PathBuf) -> Result<BackendGuard, String> {
        let python = project_root
            .join("venv")
            .join("Scripts")
            .join("python.exe");
        let backend_dir = project_root.join("backend");

        // stdout/stderr'i dosyaya yönlendir. Konsol penceresi yok ama
        // logger'a düşmeyen şeyler (native traceback, uvicorn'un erken
        // açılış hataları, çökme çıktıları) burada birikir, kaybolmaz.
        let log_path = backend_dir
            .join("data")
            .join("logs")
            .join("backend_stdout.log");
        if let Some(parent) = log_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let out = File::create(&log_path)
            .map_err(|e| format!("stdout log dosyası açılamadı ({}): {e}", log_path.display()))?;
        let err = out
            .try_clone()
            .map_err(|e| format!("stderr handle klonlanamadı: {e}"))?;

        // venv python ile: python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
        // Çalışma dizini backend/ — frontend buraya (localhost:8000) bağlanıyor.
        let child = Command::new(&python)
            .args([
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ])
            .current_dir(&backend_dir)
            .stdout(Stdio::from(out))
            .stderr(Stdio::from(err))
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|e| format!("backend başlatılamadı ({}): {e}", python.display()))?;

        unsafe {
            // Job oluştur
            let job = CreateJobObjectW(None, None)
                .map_err(|e| format!("CreateJobObject başarısız: {e}"))?;

            // KILL_ON_JOB_CLOSE bayrağını set et: job kapanınca üyeleri öldür.
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
            .map_err(|e| {
                let _ = CloseHandle(job);
                format!("SetInformationJobObject başarısız: {e}")
            })?;

            // Backend sürecini job'a ata
            let proc_handle = HANDLE(child.as_raw_handle() as *mut c_void);
            AssignProcessToJobObject(job, proc_handle).map_err(|e| {
                let _ = CloseHandle(job);
                format!("AssignProcessToJobObject başarısız: {e}")
            })?;

            Ok(BackendGuard {
                job,
                _child: child,
            })
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|_app| {
            #[cfg(windows)]
            {
                use tauri::Manager;

                // Proje kökü: bu crate frontend/src-tauri içinde, iki üst = kök.
                // Derleme-zamanı çapası (CARGO_MANIFEST_DIR) — dev için doğru.
                // NOT: Paketlemeye geçince bu yol geçersiz olur; orada backend
                // sidecar olarak resource dizininden çözülmeli.
                let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .and_then(|p| p.parent())
                    .map(|p| p.to_path_buf());

                match project_root {
                    Some(root) => match backend_proc::spawn(root) {
                        Ok(guard) => {
                            // State'e koy: uygulama ömrü boyunca canlı kalsın.
                            _app.manage(guard);
                        }
                        Err(e) => {
                            // Backend başlamazsa uygulama yine açılsın; frontend
                            // zaten "bağlanılamıyor → Tekrar Dene" gösteriyor.
                            eprintln!("[backend] başlatılamadı: {e}");
                        }
                    },
                    None => eprintln!("[backend] proje kökü çözülemedi."),
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
