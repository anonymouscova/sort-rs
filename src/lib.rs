#![feature(drain_filter)]

mod bbox;
mod process;
mod sort;
use process::regionprops;
//use redis::Commands;
use sort::Sort;
use std::net::TcpStream;
// mod track;
use std::io::*;
use serde::Serialize;

type PrecisionType = f32;

use libc::size_t;
use std::slice;

/// Wrapper around SORT tracker
pub struct Tracker {
    sort: Sort,
    socket: Option<TcpStream>,
    cc_threshold: u32,
}


impl Tracker {
    pub fn new(socket: Option<TcpStream>, height: usize, width: usize, max_age: u64, min_hits: u64, iou_threshold: f64, cc_threshold: u32, _export: bool) -> Self {
        Tracker {
            sort: Sort::new(
                      height, width, max_age, min_hits, iou_threshold as PrecisionType
                  ),
                  socket,
                  cc_threshold,
        }
    }
    pub fn update(&mut self, mask: &[u8], pts: u64) -> Option<u64> {
        let regions = regionprops(
            &mask[..self.sort.width*self.sort.height],
            self.sort.width,
            self.sort.height,
            self.cc_threshold as i32,
        ).unwrap();

        // Update
        let dead_tracks = self.sort.update(&regions, pts).unwrap();

        // Calculate the optimal timestamp that needs to be decoded
        let ret = if dead_tracks.len() != 0 {
            Some(
                dead_tracks
                .iter()
                .filter(|trk| !trk.is_seen())
                .fold(0, |max, trk| std::cmp::max(max, trk.start))
            )
        } else {
            None
        };

        if let Some(ref mut con) = self.socket {
            dead_tracks.iter().for_each(|trk| {
                trk.history.iter().for_each(|(ts, bbox)| {

            let data = SocketStruct {
                left: bbox.left,
                top: bbox.top,
                width: bbox.width,
                height: bbox.height,
                area: bbox.area,
                id: trk.id,
                ts: *ts,
            };
            con.write(serde_json::to_string(&data).unwrap().as_bytes())
                .expect("Failed to write bbox through socket");
            con.flush().unwrap();
                });

            });
        }
        ret
    }
    pub fn any(&self) -> bool {
        self.sort.any_valid()
    }

    pub fn seen(&mut self, pts: u64) {
        self.sort.mark_seen(pts);
    }
}

#[no_mangle]
pub extern "C" fn tracker_new(
    _id: u32,
    height: u32,
    width: u32,
    max_age: u64,
    min_hits: u64,
    iou_threshold: f64,
    cc_threshold: u32,
    debug: bool,
    db: u32,
) -> *mut Tracker {
    // env_logger::init();
    println!("Configuring SORT tracker with max_age: {}, min_hits: {}, iou_threshold: {}, cc_threshold: {}, debug: {}, db: {}",
        max_age, min_hits, iou_threshold, cc_threshold, debug, db );

    let socket = Some(TcpStream::connect("127.0.0.1:8989").expect("Socket Creation Failed").try_clone().unwrap());
    let ret = Box::new(Tracker::new(
            socket,
            height as usize,
            width as usize,
            max_age,
            min_hits,
            iou_threshold,
            cc_threshold,
            debug,
    ));

    Box::into_raw(ret)
}

#[no_mangle]
pub extern "C" fn tracker_update(ptr: *mut Tracker, mask: *mut u8, len: size_t, pts: u64) -> u64 {
    let tracker = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let mask = unsafe {
        assert!(!mask.is_null());
        slice::from_raw_parts_mut(mask, len as usize)
    };

    tracker.update(mask, pts).unwrap_or(0)
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize)]
pub struct SocketStruct {
    pub left: PrecisionType,
    pub top: PrecisionType,
    pub width: PrecisionType,
    pub height: PrecisionType,
    pub area: PrecisionType,
    pub id: u64,
    pub ts: u64,
}

#[no_mangle]
pub extern "C" fn tracker_any(ptr: *mut Tracker) -> bool {
    let tracker = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    tracker.any()
}

#[no_mangle]
pub extern "C" fn tracker_seen(ptr: *mut Tracker, pts: u64) {
    let tracker = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    tracker.seen(pts);
}

#[no_mangle]
pub extern "C" fn tracker_free(ptr: *mut Tracker) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(ptr);
    }
}

