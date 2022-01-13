use crate::PrecisionType;
use bytes::BytesMut;
use serde::Serialize;

#[derive(Debug, Copy, Clone, PartialEq, Serialize)]
pub struct BBox {
    pub left: PrecisionType,
    pub top: PrecisionType,
    pub width: PrecisionType,
    pub height: PrecisionType,
    pub area: PrecisionType,
}

impl BBox {
    pub fn new(
        left: PrecisionType,
        top: PrecisionType,
        width: PrecisionType,
        height: PrecisionType,
    ) -> Self {
        Self {
            left,
            top,
            width,
            height,
            area: width * height,
        }
    }

    pub fn convert_to_z(&self) -> (PrecisionType, PrecisionType, PrecisionType, PrecisionType) {
        let x = self.left + self.width / 2.;
        let y = self.top + self.height / 2.;
        let s = self.width * self.height;
        let r = self.width / self.height;

        (x, y, s, r)
    }

    /// Build BBox from &[x, y, s, r, ...]
    pub fn from_x(x: &[PrecisionType]) -> Self {
        let width = (x[2] * x[3]).sqrt();
        let height = x[2] / width;
        let left = x[0] - width / 2.;
        let top = x[1] - width / 2.;
        Self::new(left, top, width, height)
    }

    /// Return coordinate of the BBox in the form of ((x1, y1), (x2, y2))
    pub fn coordinate(
        &self,
    ) -> (
        (PrecisionType, PrecisionType),
        (PrecisionType, PrecisionType),
    ) {
        (
            (self.left, self.top),
            (self.left + self.width, self.top + self.height),
        )
    }

    pub fn iou(&self, target: &BBox) -> PrecisionType {
        let ((s_x1, s_y1), (s_x2, s_y2)) = self.coordinate();
        let ((t_x1, t_y1), (t_x2, t_y2)) = target.coordinate();

        let x_left = PrecisionType::max(s_x1, t_x1);
        let y_top = PrecisionType::max(s_y1, t_y1);
        let x_right = PrecisionType::min(s_x2, t_x2);
        let y_bottom = PrecisionType::min(s_y2, t_y2);

        if x_right <= x_left || y_bottom <= y_top {
            0.
        } else {
            let intersect_area = (x_right - x_left) * (y_bottom - y_top);
            let union_area = self.area + target.area - intersect_area;

            intersect_area / union_area
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = BytesMut::with_capacity(16);
        bytes.extend_from_slice(&(self.left as f32).to_le_bytes());
        bytes.extend_from_slice(&(self.top as f32).to_le_bytes());
        bytes.extend_from_slice(&(self.width as f32).to_le_bytes());
        bytes.extend_from_slice(&(self.height as f32).to_le_bytes());
        bytes.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_same() {
        let a = BBox::new(0., 0., 2., 2.);
        let b = BBox::new(0., 0., 2., 2.);

        let expected: PrecisionType = 1.;
        assert_eq!(a.iou(&b), expected);
    }

    #[test]
    fn test_iou_quarter() {
        let a = BBox::new(0., 0., 2., 2.);
        let b = BBox::new(1., 1., 2., 2.);

        let expected: PrecisionType = 1. / 7.;
        assert_eq!(a.iou(&b), expected);
    }

    #[test]
    fn test_iou_none() {
        let a = BBox::new(0., 0., 2., 2.);
        let b = BBox::new(2., 2., 2., 2.);

        let expected: PrecisionType = 0.;
        assert_eq!(a.iou(&b), expected);
    }

    #[test]
    fn test_to_bytes() {
        let a = BBox::new(0., 0., 2., 2.);
        println!("{:?}", a.to_bytes())
    }
}
