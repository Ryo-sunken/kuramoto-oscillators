use crate::parameter::{ControlParam, NetworkParam};
use matrix::{sparse::SparseMatrix, Matrix};
use ode_solver::{DelayedEulerSolver, EulerSolver, RungeKuttaSolver};
use std::f64::consts::PI;
use std::fs::File;

pub struct KuramotoOscillators {
    inc_trans: SparseMatrix<f64>,
    inc_wgt: SparseMatrix<f64>,
    input_wgt: SparseMatrix<f64>,
    av_wgt: Matrix<f64>,
    omega: Matrix<f64>,
    input_omega: Matrix<f64>,
    control_type: usize,
}

impl KuramotoOscillators {
    pub fn new(network_param: &NetworkParam, control_param: &ControlParam) -> Self {
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
        let inc_wgt = (&inc * &wgt).to_sparse();
        let inc_trans = inc.transpose().to_sparse();
        let input_wgt = Matrix::<f64>::from_vec_col(control_param.input_weight.clone())
            .diag()
            .to_sparse();

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
            inc_trans,
            inc_wgt,
            input_wgt,
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
            &self.omega - &self.inc_wgt * (&self.inc_trans * x).sin()
                + &self.input_wgt * (&self.av_wgt * x - x).sin()
        } else if self.control_type == 1 {
            &self.omega - &self.inc_wgt * (&self.inc_trans * x).sin()
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
            &self.omega - &self.inc_wgt * (&self.inc_trans * x).sin()
                + &self.input_wgt * (&self.av_wgt * x - x).sin()
        } else if self.control_type == 1 {
            &self.omega - &self.inc_wgt * (&self.inc_trans * x).sin()
                + &self.input_wgt * (&self.input_omega * t - x).sin()
        } else {
            x.clone()
        }
    }

    fn post_process(&self, x: &Matrix<f64>) -> Matrix<f64> {
        x.repeat(0., 2. * PI)
    }
}

pub struct DelayedKuramotoOscillators {
    inc_trans: SparseMatrix<f64>,
    inc_wgt: SparseMatrix<f64>,
    input_wgt: SparseMatrix<f64>,
    av_wgt: Matrix<f64>,
    omega: Matrix<f64>,
    input_omega: Matrix<f64>,
    control_type: usize,
    delay_buffer: Vec<Matrix<f64>>,
    delay_step: usize,
}

impl DelayedKuramotoOscillators {
    pub fn new(
        network_param: &NetworkParam,
        control_param: &ControlParam,
        delay_step: usize,
    ) -> Self {
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
        let inc_wgt = (&inc * &wgt).to_sparse();
        let inc_trans = inc.transpose().to_sparse();
        let input_wgt = Matrix::<f64>::from_vec_col(control_param.input_weight.clone())
            .diag()
            .to_sparse();

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

        let delay_buffer = Vec::with_capacity(delay_step);
        println!("{} {}", delay_step, delay_buffer.len());

        Self {
            inc_trans,
            inc_wgt,
            input_wgt,
            av_wgt,
            omega: Matrix::<f64>::from_vec_col(network_param.frequency.clone()),
            input_omega: Matrix::<f64>::from_vec_col(control_param.input_frequency.clone()),
            control_type: control_param.control_type,
            delay_buffer,
            delay_step,
        }
    }
}

impl DelayedEulerSolver<f64, File> for DelayedKuramotoOscillators {
    fn dot_x(&self, x: &Matrix<f64>, t: f64, k: usize) -> Matrix<f64> {
        if self.control_type == 0 {
            if k < self.delay_step {
                &self.omega - &self.inc_wgt * (&self.inc_trans * x).sin()
            } else {
                let delayed_phase = &self.delay_buffer[k % self.delay_step];
                &self.omega - &self.inc_wgt * (&self.inc_trans * x).sin()
                    + &self.input_wgt * (&self.av_wgt * delayed_phase - x).sin()
            }
        } else if self.control_type == 1 {
            &self.omega - &self.inc_wgt * (&self.inc_trans * x).sin()
                + &self.input_wgt * (&self.input_omega * t - x).sin()
        } else {
            x.clone()
        }
    }

    fn post_process(&self, x: &Matrix<f64>) -> Matrix<f64> {
        x.repeat(0., 2. * PI)
    }

    fn push_buffer(&mut self, x: &Matrix<f64>, k: usize) {
        if k < self.delay_step {
            self.delay_buffer.push(x.clone());
        }
        else {
            self.delay_buffer[k % self.delay_step] = x.clone();
        }
    }
}
