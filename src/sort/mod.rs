mod tracker;
use crate::bbox::BBox;
use crate::PrecisionType;
use linear_assignment::{solver, MatrixSize};
use nalgebra::DMatrix;
use ordered_float::OrderedFloat;
use tracker::KalmanBoxTracker as Tracker;

pub struct Sort {
    pub width: usize,
    pub height: usize,
    pub max_age: u64,
    min_hits: u64,
    iou_threshold: PrecisionType,
    pub trackers: Vec<Tracker>,
    frame_count: u64,
    unique_id: u64,
}

fn linear_assignment(cost_matrix: &DMatrix<OrderedFloat<PrecisionType>>) -> Vec<(usize, usize)> {
    let (n_trackers, n_dets) = cost_matrix.shape();

    let mut target = if n_trackers != n_dets {
        let longer = std::cmp::max(n_trackers, n_dets);
        let extended_size = longer * longer;
        let mut zero_vec = Vec::with_capacity(extended_size);
        for _ in 0..extended_size {
            zero_vec.push(OrderedFloat(0.0));
        }
        let mut extended_costs = DMatrix::from_vec(longer, longer, zero_vec);
        extended_costs
            .slice_mut((0, 0), (n_trackers, n_dets))
            .copy_from(cost_matrix);
        extended_costs
    } else {
        cost_matrix.clone()
    };
    let size = MatrixSize {
        rows: target.nrows(),
        columns: target.ncols(),
    };

    let max_weight = OrderedFloat(2.);

    let edges = solver(&mut target, &size);
    edges
        .into_iter()
        .filter(|(i, j)| i < &n_trackers && j < &n_dets)
        .filter(|(i, j)| cost_matrix[(*i, *j)] != max_weight)
        .collect()
}

impl Sort {
    pub fn new(
        width: usize,
        height: usize,
        max_age: u64,
        min_hits: u64,
        iou_threshold: PrecisionType,
    ) -> Self {
        Sort {
            width,
            height,
            max_age,
            min_hits,
            iou_threshold,
            trackers: vec![],
            frame_count: 0,
            unique_id: 0,
        }
    }

    /// # Generate cost matrix of negated IoU
    ///
    /// Note that the matrix is in the form of track => detection
    ///     det0 det1
    /// trk0
    /// trk1
    fn generate_cost_matrix(
        &self,
        dets: &Vec<BBox>,
    ) -> Result<DMatrix<OrderedFloat<PrecisionType>>, anyhow::Error> {
        let iou_threshold = self.iou_threshold;
        let n_trackers = self.trackers.len();
        let n_dets = dets.len();

        let cost_matrix = self
            .trackers
            .iter()
            .map(|trk| (trk.prior(), trk.active))
            .flat_map(|(pos, active)| dets.into_iter().map(move |det| (pos.iou(det), active)))
            .map(|(iou, active)| {
                if iou < iou_threshold {
                    OrderedFloat(2.)
                } else {
                    if active {
                        // Prefer active tracks
                        OrderedFloat(1. - iou)
                    } else {
                        OrderedFloat(2. - iou)
                    }
                }
            })
        .collect::<Vec<OrderedFloat<PrecisionType>>>();
        Ok(DMatrix::from_vec(n_dets, n_trackers, cost_matrix).transpose())
    }

    /// Match detections to existing trackers
    /// and returns (tracker, detection) match
    /// Solve assignment problem on the cost matrix
    fn match_dets(&self, dets: &Vec<BBox>) -> Result<Vec<(usize, usize)>, anyhow::Error> {
        let n_trackers = self.trackers.len();

        Ok(if n_trackers > 0 {
            let cost_matrix = self.generate_cost_matrix(dets)?;
            let assigned = linear_assignment(&cost_matrix);
            assigned
                .into_iter()
                .filter(|(i, j)| {
                    let threshold = if self.trackers[*i].active {
                        OrderedFloat(1. - self.iou_threshold)
                    } else {
                        OrderedFloat(2. - self.iou_threshold)
                    };
                    cost_matrix[(*i, *j)] <= threshold
                })
            .collect()
                // let negated_threshold = OrderedFloat(1. - self.iou_threshold);
        } else {
            vec![]
        })
    }

    /// Execute `predict` for every trackers
    fn predict(&mut self, ts: u64) {
        self.trackers.iter_mut().for_each(|trk| {
            trk.predict(ts);
            ()
        });
    }

    /// Perform tracking on the new frame and returns the least PTS of unseen object
    pub fn update(&mut self, dets: &Vec<BBox>, pts: u64) -> Result<Vec<Tracker>, anyhow::Error> {
        self.frame_count += 1;
        let n_dets = dets.len();

        self.predict(pts);
        let matches = self.match_dets(dets)?;
        let unmatched_det_iter = (0..n_dets).filter(|i| matches.iter().all(|(_, j)| i != j));
        let unmatched_trk_iter = (0..self.trackers.len()).filter(|i| matches.iter().all(|(j, _)| i != j));
        unmatched_trk_iter.for_each(|i| self.trackers[i].hit_streaks = 0);

        // Update matched trackers with assigned detections
        for (track_idx, det_idx) in matches.iter() {
            self.trackers[*track_idx].update(&dets[*det_idx], pts)?;
        }

        // Activate tracks older than self.min_hits
        let min_hits = self.min_hits;
        self.trackers
            .iter_mut()
            .for_each(|trk| trk.check_activate(min_hits));

        let max_age = self.max_age;

        let dead_tracks = self
            .trackers
            .drain_filter(|trk| !trk.should_live(max_age))
            .filter(|trk| trk.active)
            .map(|mut trk| { trk.trim_dead_history(); trk} )
            .collect();

        // Initialize new trackers for unmatched detections
        for i in unmatched_det_iter {
            let tracker = Tracker::new(self.unique_id, &dets[i], pts);
            self.unique_id += 1;
            self.trackers.push(tracker);
        }

        Ok(dead_tracks)
    }

    /// Return bounding box of tracks at certain timestamp
    pub fn tracks_at(&self, ts: u64) -> Vec<(u64, BBox)> {
        self.trackers
            .iter()
            .filter_map(|trk| trk.location_at(ts).map(|bbox| (trk.id, bbox)))
            .collect()
    }

    pub fn active_at(&self, ts: u64) -> Vec<(u64, BBox)> {
        self.trackers
            .iter()
            .filter(|trk| trk.active)
            .filter_map(|trk| trk.location_at(ts).map(|bbox| (trk.id, bbox)))
            .collect()
    }

    pub fn mark_seen(&mut self, ts: u64) {
        self.trackers
            .iter_mut()
            .for_each(|trk| trk.seen_ts.push(ts));
    }

    pub fn mark_active_seen(&mut self, ts: u64) {
        self.trackers
            .iter_mut()
            .filter(|trk| trk.active)
            .filter(|trk| trk.start <= ts)
            .for_each(|trk| trk.seen_ts.push(ts));
    }

    pub fn any_valid(&self) -> bool {
        self.trackers.iter().any(|trk| trk.active)
    }
}

impl Default for Sort {
    fn default() -> Self {
        let width = 80 * 2;
        let height = 45 * 2;
        let max_age = 3;
        let min_hits = 3;
        let iou_threshold = 0.2;
        Sort::new(width, height, max_age, min_hits, iou_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_sort() -> anyhow::Result<()> {
        let mut sort: Sort = Default::default();
        let dets = vec![BBox::new(0., 0., 2., 2.), BBox::new(1., 1., 2., 2.)];
        sort.update(&dets, 0)?;

        // Check fields
        assert_eq!(sort.frame_count, 1);

        // Check trackers
        assert_eq!(sort.trackers.len(), 2);
        let states = sort.trackers.iter().map(|trk| trk.get_state());
        for (i, state) in states.enumerate() {
            assert_eq!(state, dets[i]);
        }
        Ok(())
    }
    #[test]
    #[should_panic]
    fn test_prior_before_predict() {
        let mut sort: Sort = Default::default();
        let dets = vec![BBox::new(0., 0., 2., 2.), BBox::new(1., 1., 2., 2.)];
        // Initialize new trackers
        sort.update(&dets, 0).unwrap();
        assert_eq!(sort.trackers.len(), 2);
        // Accessing prior without calling predict should panic
        let _states: Vec<_> = sort
            .trackers
            .iter()
            .map(|trk| trk.prior().clone())
            .collect();
    }
    #[test]
    fn test_obeservation_model() -> anyhow::Result<()> {
        let mut sort: Sort = Default::default();
        let dets = vec![BBox::new(0., 0., 2., 2.), BBox::new(1., 1., 2., 2.)];
        // Initialize new trackers
        sort.update(&dets, 0)?;
        assert_eq!(sort.trackers.len(), 2);
        // Perform prediction
        sort.predict(0);

        // Check trackers
        assert_eq!(sort.trackers.len(), 2);
        let states = sort.trackers.iter().map(|trk| trk.prior().clone());
        for (i, state) in states.enumerate() {
            assert_eq!(state, dets[i]);
        }
        Ok(())
    }
    fn expected_iou(
        b1: &BBox,
        b2: &BBox,
        iou_threshold: PrecisionType,
    ) -> OrderedFloat<PrecisionType> {
        let iou = b1.iou(b2);

        if iou >= iou_threshold {
            OrderedFloat(2. - iou)
        } else {
            OrderedFloat(2.)
        }
    }

    #[test]
    fn test_generate_cost_matrix_same() -> anyhow::Result<()> {
        let mut sort: Sort = Default::default();
        let dets = vec![BBox::new(0., 0., 4., 4.), BBox::new(1., 1., 4., 4.)];
        // Initialize new trackers
        sort.update(&dets, 0)?;
        assert_eq!(sort.trackers.len(), 2);
        // Perform prediction
        sort.predict(0);

        // Generate with exact same bboxes
        let cost_matrix = sort.generate_cost_matrix(&dets)?;
        let expected: DMatrix<OrderedFloat<PrecisionType>> = DMatrix::from_vec(
            2,
            2,
            vec![
            expected_iou(&dets[0], &dets[0], sort.iou_threshold),
            expected_iou(&dets[0], &dets[1], sort.iou_threshold),
            expected_iou(&dets[1], &dets[0], sort.iou_threshold),
            expected_iou(&dets[1], &dets[1], sort.iou_threshold),
            ],
        );
        assert_eq!(cost_matrix, expected);

        Ok(())
    }

    #[test]
    fn test_generate_cost_matrix() -> anyhow::Result<()> {
        let mut sort: Sort = Default::default();
        let first_dets = vec![BBox::new(0., 0., 4., 4.), BBox::new(1., 1., 4., 4.)];
        // Initialize new trackers
        sort.update(&first_dets, 0)?;
        assert_eq!(sort.trackers.len(), 2);
        // Perform prediction
        sort.predict(0);

        let second_dets = vec![
            BBox::new(1., 1., 4., 4.),
            BBox::new(2., 2., 4., 4.),
            BBox::new(3., 3., 4., 4.),
        ];
        // Generate with exact same bboxes
        let cost_matrix = sort.generate_cost_matrix(&second_dets)?;
        let expected = DMatrix::from_vec(
            2,
            3,
            vec![
            expected_iou(&first_dets[0], &second_dets[0], sort.iou_threshold),
            expected_iou(&first_dets[1], &second_dets[0], sort.iou_threshold),
            expected_iou(&first_dets[0], &second_dets[1], sort.iou_threshold),
            expected_iou(&first_dets[1], &second_dets[1], sort.iou_threshold),
            expected_iou(&first_dets[0], &second_dets[2], sort.iou_threshold),
            expected_iou(&first_dets[1], &second_dets[2], sort.iou_threshold),
            ],
        );
        assert_eq!(cost_matrix, expected);

        Ok(())
    }
    #[test]
    fn test_linear_assignment_5x5() {
        let sort: Sort = Default::default();
        #[rustfmt::skip]
        let cost_matrix: DMatrix<OrderedFloat<PrecisionType>>  = DMatrix::from_vec(
            5, 5,
            vec! [
            -1.,  0., 0.,  0., 0.,
            0., -1., 0.,  0., 0.,
            0.,  0., 0., -1., 0.,
            0.,  0., 0.,  0., 0.,
            0.,  0., 0.,  0., 0.,
            ].into_iter().map(|x| OrderedFloat(2. + x)).collect::<Vec<OrderedFloat<PrecisionType>>>()
        );

        let mut result = linear_assignment(&cost_matrix);
        #[rustfmt::skip]
        let mut expected = vec![
            (0, 0), (1, 1), (3, 2)
        ];

        result.sort();
        expected.sort();
        assert_eq!(result, expected);
    }
    #[test]
    fn test_linear_assignment_2x3() {
        let sort: Sort = Default::default();

        #[rustfmt::skip]
        let cost_matrix = DMatrix::from_vec(
            2, 3,
            vec! [
            -1.,  0.,
            0.,  0.,
            0., -1.,
            ].into_iter().map(|x| OrderedFloat(1. + x)).collect::<Vec<OrderedFloat<PrecisionType>>>()
        );

        let mut result = linear_assignment(&cost_matrix);
        #[rustfmt::skip]
        let mut expected = vec![
            (0, 0), (1, 2)
        ];

        result.sort();
        expected.sort();
        assert_eq!(result, expected);
    }
    #[test]
    fn test_linear_assignment_3x2() {
        let sort: Sort = Default::default();

        #[rustfmt::skip]
        let cost_matrix = DMatrix::from_vec(
            3, 2,
            vec! [
            -1., 0.,  0.,
            0., 0., -1.,
            ].into_iter().map(|x| OrderedFloat(1. + x)).collect::<Vec<OrderedFloat<PrecisionType>>>()
        );

        let mut result = linear_assignment(&cost_matrix);
        #[rustfmt::skip]
        let mut expected = vec![
            (0, 0), (2, 1)
        ];

        result.sort();
        expected.sort();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_linear_assignment_9x8() {
        let sort: Sort = Default::default();

        #[rustfmt::skip]
        let cost_matrix = DMatrix::from_vec(
            9, 8,
            vec! [
            -1.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,
            0., -1.,  0., 0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0., -1., 0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0., 0., -1.,  0.,  0.,  0.,  0.,
            0.,  0.,  0., 0.,  0., -1.,  0.,  0.,  0.,
            0.,  0.,  0., 0.,  0.,  0., -1.,  0.,  0.,
            0.,  0.,  0., 0.,  0.,  0.,  0., -1.,  0.,
            0.,  0.,  0., 0.,  0.,  0.,  0.,  0., -1.,
            ].into_iter().map(|x| OrderedFloat(1. + x)).collect::<Vec<OrderedFloat<PrecisionType>>>()
        );

        let mut result = linear_assignment(&cost_matrix);
        #[rustfmt::skip]
        let mut expected = vec![
            (0, 0), (1, 1), (2, 2), (4, 3),
            (5, 4), (6, 5), (7, 6), (8, 7),
        ];

        result.sort();
        expected.sort();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_match_dets() -> anyhow::Result<()> {
        let mut sort: Sort = Default::default();
        let first_dets = vec![BBox::new(0., 0., 4., 4.), BBox::new(1., 1., 4., 4.)];
        // Initialize new trackers
        sort.update(&first_dets, 0)?;
        assert_eq!(sort.trackers.len(), 2);
        // Perform prediction
        sort.predict(0);

        let second_dets = vec![
            BBox::new(1., 1., 4., 4.),
            BBox::new(2., 2., 4., 4.),
            BBox::new(3., 3., 4., 4.),
        ];
        // Generate with exact same bboxes
        let matches = sort.match_dets(&second_dets)?;
        let expected = vec![(1, 0)];
        assert_eq!(matches, expected);
        Ok(())
    }
}
