use std::{fs, io};
use rand_chacha::ChaCha8Rng;

pub fn create_param_dirs(name: &str) -> io::Result<()> {
    fs::create_dir(format!("data/{}", name))?;
    fs::create_dir(format!("data/{}/param", name))?;
    fs::create_dir(format!("data/{}/param/network", name))?;
    fs::create_dir(format!("data/{}/param/control", name))?;
    fs::create_dir(format!("data/{}/result", name))?;

    Ok(())
}

pub fn watts_strogatz_model(n: usize, k: usize, p: f64, rng: &mut ChaCha8Rng) {

}
