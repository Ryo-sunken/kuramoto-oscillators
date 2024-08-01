mod param_generator;
mod parameter;
mod kuramoto_oscillators;

use matrix::Matrix;
use ode_solver::{EulerSolver, RungeKuttaSolver};
use parameter::{CommonParam, ControlParam, NetworkParam};
use rand_chacha::rand_core::SeedableRng;
use std::path::Path;
use std::{fs, io};
use crate::kuramoto_oscillators::KuramotoOscillators;

const DIR_NAME: &str = "toy1";
const IS_GEN_PARAM: bool = false;

fn get_file_stem(name: &str) -> &str {
    let split_name: Vec<&str> = name.split('.').collect();
    split_name[0]
}

fn read_dir<P: AsRef<Path>>(path: P) -> io::Result<Vec<String>> {
    Ok(fs::read_dir(path)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            if entry.file_type().ok()?.is_file() {
                Some(entry.file_name().to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect())
}

fn initialize(n: usize, random_range: f64, seed: u64) -> Matrix<f64> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    //Matrix::one(n, 1)
    Matrix::randu(n, 1, &mut rng) * 2.0 * std::f64::consts::PI * random_range
}

fn main() {
    println!("start");

    // パラメータの生成（オプション）
    if IS_GEN_PARAM {
        let _ = param_generator::create_param_dirs(DIR_NAME);
    }

    // 共通パラメータの読み込み
    let common_param = CommonParam::load().unwrap();

    // パラメータファイル一覧の取得
    let network_param_names = read_dir(format!("data/{}/param/network", DIR_NAME)).unwrap();
    let control_param_names = read_dir(format!("data/{}/param/control", DIR_NAME)).unwrap();

    // シミュレーションの実行
    for network in &network_param_names {
        let _ = fs::create_dir(format!(
            "data/{}/result/{}",
            DIR_NAME,
            get_file_stem(network)
        ));
        for control in &control_param_names {
            println!("{}, {}", network, control);

            // 結果を保存するディレクトリの作成
            let network_name = get_file_stem(network);
            let control_name = get_file_stem(control);
            let dir_path = format!("data/{}/result/{}/{}", DIR_NAME, network_name, control_name);
            let _ = fs::create_dir(&dir_path);

            // パラメータの読み込み
            let network_param =
                NetworkParam::load(&format!("data/{}/param/network/{}", DIR_NAME, network))
                    .unwrap();
            let control_param =
                ControlParam::load(&format!("data/{}/param/control/{}", DIR_NAME, control)).unwrap();
            let kuramoto_osc = KuramotoOscillators::new(&network_param, &control_param);

            for seed in &common_param.random_seeds {
                println!("{}", seed);

                // 初期値と結果を保存するファイル名
                let x = initialize(network_param.state_dim, common_param.random_range, *seed);
                let result_file_path = format!("{}/{}.csv", &dir_path, seed);
                let mut result_file = csv::WriterBuilder::new()
                    .delimiter(b' ')
                    .from_path(&result_file_path)
                    .unwrap();

                RungeKuttaSolver::solve(
                    &kuramoto_osc,
                    x,
                    common_param.dt,
                    common_param.simulation_time,
                    &mut result_file,
                );
            }
        }
    }

    // 結果画像の作成（Tikz）

    println!("finish");
}
