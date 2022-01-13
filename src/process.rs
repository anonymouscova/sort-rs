use crate::bbox::BBox;
use crate::PrecisionType;
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::region_labelling::{connected_components, Connectivity};
use opencv::core::MatTraitManual;
use opencv::imgproc::{connected_components_with_stats, ConnectedComponentsTypes};
use opencv::prelude::*;

#[warn(dead_code)]
fn get_connected(image: &GrayImage) -> ImageBuffer<Luma<u32>, Vec<u32>> {
    const BG_COLOR: Luma<u8> = Luma([0u8]);
    connected_components(image, Connectivity::Four, BG_COLOR)
}

#[warn(dead_code)]
fn get_connected_cv(image: &GrayImage) -> opencv::Result<ImageBuffer<Luma<u32>, Vec<u32>>> {
    let width = image.width() as i32;
    let height = image.height() as i32;

    let input = Mat::from_slice(image.as_raw())?;
    let input = input.reshape_nd(1, &[width, height])?;

    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let connectivity = 4;
    let ltype = opencv::core::CV_32S;

    let _num_objects = connected_components_with_stats(
        &input,
        &mut labels,
        &mut stats,
        &mut centroids,
        connectivity,
        ltype,
    )?;

    let labels: Vec<_> = labels.to_vec_2d::<i32>()?;

    Ok(ImageBuffer::from_raw(
        width as u32,
        height as u32,
        labels
            .into_iter()
            .flatten()
            .map(|x| x as u32)
            .collect::<Vec<_>>(),
    )
    .unwrap())
}

pub fn regionprops(
    raw_slice: &[u8],
    width: usize,
    height: usize,
    area_thresh: i32,
) -> opencv::Result<Vec<BBox>> {
    let width = width as i32;
    let height = height as i32;

    let input = Mat::from_slice(raw_slice)?;
    // let input = input.reshape_nd(1, &[height, width])?;
    let input = input.reshape_nd(1, &[width, height])?;

    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let connectivity = 8;
    let ltype = opencv::core::CV_32S;

    let num_objects = connected_components_with_stats(
        // FIXME: This swaps left & top
        &input,
        &mut labels,
        &mut stats,
        &mut centroids,
        connectivity,
        ltype,
    )?;

    const LEFT: i32 = ConnectedComponentsTypes::CC_STAT_LEFT as i32;
    const TOP: i32 = ConnectedComponentsTypes::CC_STAT_TOP as i32;
    const WIDTH: i32 = ConnectedComponentsTypes::CC_STAT_WIDTH as i32;
    const HEIGHT: i32 = ConnectedComponentsTypes::CC_STAT_HEIGHT as i32;
    const AREA: i32 = ConnectedComponentsTypes::CC_STAT_AREA as i32;

    let mut ret = vec![];

    for i in 1..num_objects {
        if stats.at_2d::<i32>(i, AREA)? < &area_thresh {
            continue;
        }

        let &left = stats.at_2d::<i32>(i, LEFT)?;
        let &top = stats.at_2d::<i32>(i, TOP)?;
        let &width = stats.at_2d::<i32>(i, WIDTH)?;
        let &height = stats.at_2d::<i32>(i, HEIGHT)?;

        ret.push(BBox::new(
            left as PrecisionType,
            top as PrecisionType,
            width as PrecisionType,
            height as PrecisionType,
        ));
    }

    Ok(ret)
}

#[cfg(test)]
mod test {
    // use image::{GrayImage, ImageBuffer, Luma};
    use super::*;
    #[test]
    fn test_connected_component() {
        #[rustfmt::skip]
        let image = GrayImage::from_raw(
            5,
            4,
            vec![
                1, 0, 1, 1, 0,
                0, 1, 1, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 1, 0
            ],
        )
        .unwrap();

        #[rustfmt::skip]
        let expected: ImageBuffer<Luma<u32>, Vec<u32>> = ImageBuffer::from_raw(
            5,
            4,
            vec![
                1, 0, 2, 2, 0,
                0, 2, 2, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 3, 0
            ],
        )
        .unwrap();

        assert_eq!(get_connected(&image), expected);
        assert_eq!(get_connected_cv(&image).unwrap(), expected);
    }
}
