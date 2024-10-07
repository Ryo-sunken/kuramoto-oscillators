mod kuramoto_oscillators;
mod param_generator;
mod parameter;

use crate::kuramoto_oscillators::KuramotoOscillators;
use kuramoto_oscillators::DelayedKuramotoOscillators;
use matrix::Matrix;
use mpi::traits::*;
use ode_solver::{DelayedEulerSolver, EulerSolver, RungeKuttaSolver};
use parameter::{CommonParam, ControlParam, NetworkParam};
use rand_chacha::rand_core::SeedableRng;
use std::path::Path;
use std::{fs, io};

const DIR_NAME: &str = "report1";

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
    Matrix::randu(n, 1, &mut rng) * 2.0 * std::f64::consts::PI * random_range
}

fn normal_simulation() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    if rank == 0 {
        println!("start");
    }

    // 共通パラメータの読み込み
    let common_param = CommonParam::from_path("data/common.json").unwrap();

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
            if rank == 0 {
                println!("{}, {}", network, control);
            }

            // 結果を保存するディレクトリの作成
            let network_name = get_file_stem(network);
            let control_name = get_file_stem(control);
            let dir_path = format!("data/{}/result/{}/{}", DIR_NAME, network_name, control_name);
            let _ = fs::create_dir(&dir_path);

            // パラメータの読み込み
            let network_param =
                NetworkParam::from_path(&format!("data/{}/param/network/{}", DIR_NAME, network));
            let control_param =
                ControlParam::from_path(&format!("data/{}/param/control/{}", DIR_NAME, control));
            let kuramoto_osc = KuramotoOscillators::new(&network_param, &control_param);

            let seed = common_param.random_seeds[rank as usize];
            println!("{}", seed);

            // 初期値と結果を保存するファイル名
            let x = initialize(network_param.state_dim, common_param.random_range, seed);
            let result_file_path = format!("{}/{}.csv", &dir_path, seed);
            let mut result_file = csv::WriterBuilder::new()
                .delimiter(b' ')
                .from_path(&result_file_path)
                .unwrap();

            RungeKuttaSolver::solve(
                &kuramoto_osc,
                &x,
                common_param.dt,
                common_param.simulation_time,
                &mut result_file,
            );
        }
    }

    if rank == 0 {
        println!("finish");
    }
}

fn delayed_simulation() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    if rank == 0 {
        println!("start");
    }

    // 共通パラメータの読み込み
    let common_param = CommonParam::from_path("data/common.json").unwrap();

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
            if rank == 0 {
                println!("{}, {}", network, control);
            }

            let seed = common_param.random_seeds[rank as usize];
            println!("{}", seed);

            // 結果を保存するディレクトリの作成
            let network_name = get_file_stem(network);
            let control_name = get_file_stem(control);
            let dir_path = format!(
                "data/{}/result/{}/delayed_{}/{}",
                DIR_NAME, network_name, control_name, seed
            );
            let _ = fs::create_dir(&dir_path);

            // パラメータの読み込み
            let network_param =
                NetworkParam::from_path(&format!("data/{}/param/network/{}", DIR_NAME, network));
            let control_param =
                ControlParam::from_path(&format!("data/{}/param/control/{}", DIR_NAME, control));

            for delay_step in 1..100 {
                let mut kuramoto_osc =
                    DelayedKuramotoOscillators::new(&network_param, &control_param, delay_step);
                let x = initialize(network_param.state_dim, common_param.random_range, seed);
                let result_file_path = format!("{}/{}.csv", &dir_path, delay_step);
                let mut result_file = csv::WriterBuilder::new()
                    .delimiter(b' ')
                    .from_path(&result_file_path)
                    .unwrap();

                DelayedEulerSolver::solve(
                    &mut kuramoto_osc,
                    &x,
                    common_param.dt,
                    common_param.simulation_time,
                    &mut result_file,
                );
            }
        }
    }

    if rank == 0 {
        println!("finish");
    }
}

fn main() {
    delayed_simulation();
}
