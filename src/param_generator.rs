use matrix::Matrix;
use rand_chacha::ChaCha8Rng;
use std::{fs, io};

pub fn create_param_dirs(name: &str) -> io::Result<()> {
    fs::create_dir(format!("data/{}", name))?;
    fs::create_dir(format!("data/{}/param", name))?;
    fs::create_dir(format!("data/{}/param/network", name))?;
    fs::create_dir(format!("data/{}/param/control", name))?;
    fs::create_dir(format!("data/{}/result", name))?;

    Ok(())
}

#[allow(dead_code)]
pub fn watts_strogatz_model(n: usize, k: usize, p: f64, rng: &mut ChaCha8Rng) -> Matrix<usize> {
    let mut adj = Matrix::zero(n, n);
    for i in 0..n {
        for j in 1..k {
            adj[i][(i + j) % n] = 1;
            adj[i][(i - j) % n] = 1;
        }
    }
    adj
}
