mod param_generator;
mod parameter;

use matrix::Matrix;
use ode_solver::{EulerSolver, RungeKuttaSolver};
use parameter::{CommonParam, ControlParam, NetworkParam};
use rand_chacha::rand_core::SeedableRng;
use serde_json::error::Error;
use std::f64::consts::PI;
use std::fs::File;
use std::path::Path;
use std::{fs, io};

const DIR_NAME: &str = "toy1";
const IS_GEN_PARAM: bool = false;

struct KuramotoOscillators {
    inc: Matrix<f64>,
    wgt: Matrix<f64>,
    input_wgt: Matrix<f64>,
    av_wgt: Matrix<f64>,
    omega: Matrix<f64>,
    input_omega: Matrix<f64>,
    control_type: usize,
}

impl KuramotoOscillators {
    fn new(network_param: &NetworkParam, control_param: &ControlParam) -> Self {
        let edges = network_param
            .connectivity
            .iter()
            .flatten()
            .filter(|&x| *x != 0.)
            .count()
            / 2;
        let mut inc = Matrix::<f64>::zero(network_param.state_dim, edges);
        let mut wgt = Matrix::<f64>::zero(edges, edges);
        let mut av_wgt = Matrix::<f64>::zero(network_param.state_dim, network_param.state_dim);

        let mut edge_count = 0;
        for i in 0..network_param.state_dim {
            for j in i..network_param.state_dim {
                if network_param.connectivity[i][j] != 0. {
                    inc[i][edge_count] = 1.;
                    inc[j][edge_count] = -1.;
                    wgt[edge_count][edge_count] = network_param.connectivity[i][j];
                    edge_count += 1;
                }
            }
        }

        let mut start = 0;
        for k in 0..network_param.cluster_nodes_num.len() {
            let sum: f64 = control_param.average_weight[k].iter().sum();
            for i in 0..network_param.cluster_nodes_num[k] {
                for j in 0..network_param.cluster_nodes_num[k] {
                    av_wgt[i + start][j + start] = control_param.average_weight[k][j] / sum;
                }
            }
            start += network_param.cluster_nodes_num[k];
        }

        Self {
            inc,
            wgt,
            input_wgt: Matrix::<f64>::from_vec_col(control_param.input_weight.clone()).diag(),
            av_wgt,
            omega: Matrix::<f64>::from_vec_col(network_param.frequency.clone()),
            input_omega: Matrix::<f64>::from_vec_col(control_param.input_frequency.clone()),
            control_type: control_param.control_type,
        }
    }
}

impl EulerSolver<f64, File> for KuramotoOscillators {
    fn dot_x(&self, x: &Matrix<f64>, t: f64) -> Matrix<f64> {
        if self.control_type == 0 {
            &self.omega - &self.inc * &self.wgt * (&self.inc.transpose() * x).sin()
                + &self.input_wgt * (&self.av_wgt * x - x).sin()
        } else if self.control_type == 1 {
            &self.omega - &self.inc * &self.wgt * (&self.inc.transpose() * x).sin()
                + &self.input_wgt * (&self.input_omega * t - x).sin()
        } else {
            x.clone()
        }
    }

    fn post_process(&self, x: &Matrix<f64>) -> Matrix<f64> {
        x.repeat(0., 2. * PI)
    }
}

impl RungeKuttaSolver<f64, File> for KuramotoOscillators {
    fn dot_x(&self, x: &Matrix<f64>, t: f64) -> Matrix<f64> {
        if self.control_type == 0 {
            &self.omega - &self.inc * &self.wgt * (&self.inc.transpose() * x).sin()
                + &self.input_wgt * (&self.av_wgt * x - x).sin()
        } else if self.control_type == 1 {
            &self.omega - &self.inc * &self.wgt * (&self.inc.transpose() * x).sin()
                + &self.input_wgt * (&self.input_omega * t - x).sin()
        } else {
            x.clone()
        }
    }

    fn post_process(&self, x: &Matrix<f64>) -> Matrix<f64> {
        x.repeat(0., 2. * PI)
    }
}

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

fn load_common_param() -> Result<CommonParam, Error> {
    serde_json::from_str(&fs::read_to_string("data/common.json").unwrap())
}

fn load_network_param(name: &str) -> Result<NetworkParam, Error> {
    serde_json::from_str(
        &fs::read_to_string(format!("data/{}/param/network/{}", DIR_NAME, name)).unwrap(),
    )
}

fn load_control_param(name: &str) -> Result<ControlParam, Error> {
    serde_json::from_str(
        &fs::read_to_string(format!("data/{}/param/control/{}", DIR_NAME, name)).unwrap(),
    )
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
        param_generator::create_param_dirs(DIR_NAME).expect("Make Dir Failed.");
    }

    // 共通パラメータの読み込み
    let common_param = load_common_param().unwrap();

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
            let network_param = load_network_param(network).unwrap();
            let control_param = load_control_param(control).unwrap();
            let kuramoto_osc = KuramotoOscillators::new(&network_param, &control_param);

            for seed in &common_param.random_seeds {
                println!("{}", seed);

                // 初期値と結果を保存するファイル名
                let x = initialize(network_param.state_dim, common_param.random_range, *seed);
                let result_file_path = format!("{}/{}.csv", &dir_path, seed);
                let mut result_file = csv::WriterBuilder::new().delimiter(b' ').from_path(&result_file_path).unwrap();

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
