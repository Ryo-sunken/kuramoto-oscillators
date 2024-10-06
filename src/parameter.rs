use serde::{Deserialize, Serialize};
use serde_json::error::Error;
use std::fs;

#[derive(Serialize, Deserialize, Debug)]
pub struct CommonParam {
    pub dt: f64,
    pub simulation_time: f64,
    pub random_range: f64,
    pub random_seeds: Vec<u64>,
}

impl CommonParam {
    pub fn from_path(path: &str) -> Result<Self, Error> {
        serde_json::from_str(&fs::read_to_string(path).unwrap())
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NetworkParam {
    comment: String,
    pub state_dim: usize,
    pub cluster_nodes_num: Vec<usize>,
    pub frequency: Vec<f64>,
    pub connectivity: Vec<Vec<f64>>,
}

impl NetworkParam {
    pub fn from_path(path: &str) -> Self {
        let param: Self = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        param.check();
        param
    }

    fn check(&self) {
        // shape
        assert_eq!(self.cluster_nodes_num.iter().sum::<usize>(), self.state_dim);
        assert_eq!(self.frequency.len(), self.state_dim);
        assert_eq!(self.connectivity.len(), self.state_dim);
        for i in 0..self.connectivity.len() {
            assert_eq!(self.connectivity[i].len(), self.state_dim);
        }

        // connectivityが非負で対称
        assert!(self.connectivity.iter().flatten().all(|&x| x >= 0.));
        for i in 0..self.state_dim {
            for j in i..self.state_dim {
                if self.connectivity[i][j] != self.connectivity[j][i] {
                    panic!("connectivity is not symmetric.");
                }
            }
        }

    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ControlParam {
    comment: String,
    pub control_type: usize,
    pub input_dim: usize,
    pub cluster_nodes_num: Vec<usize>,
    pub input_weight: Vec<f64>,
    pub input_frequency: Vec<f64>,
    pub average_weight: Vec<Vec<f64>>,
}

impl ControlParam {
    pub fn from_path(path: &str) -> Self {
        let param: Self = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        param.check();
        param
    }

    fn check(&self) {
        // shape
        assert_eq!(self.cluster_nodes_num.iter().sum::<usize>(), self.input_dim);
        assert_eq!(self.input_weight.len(), self.input_dim);
        assert_eq!(self.input_frequency.len(), self.input_dim);
        assert_eq!(self.average_weight.iter().map(|arr| arr.iter().count()).collect::<Vec<_>>(), self.cluster_nodes_num);

        // weightが非負
        assert!(self.input_weight.iter().all(|&x| x >= 0.));
        assert!(self.average_weight.iter().flatten().all(|&x| x >= 0.));
    }
}
