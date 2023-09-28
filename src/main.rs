extern crate image;
use image::GrayImage;
use image::io::Reader as ImageReader;
use std::cmp::max;

const HARRIS_CONST: f64 = 0.04;

const R_THRESHOLD: f64 = 100000000f64;

const MAX_WINDOW_SIZE: (usize, usize) = (5, 5);

fn main() {
    let img = ImageReader::open(std::env::args().nth(1).unwrap()).unwrap().decode().unwrap();
    //let edges = detect_edges(&img.into_luma8());
    //edges.save("realedges.png").unwrap();
    let blurred_img = blur(&img.into_luma8());
    let corners = detect_corners(&blurred_img);
    corners.save("corners.png").unwrap();
    real_coords((100f64, 100f64), (1f64, 1f64), (-50f64, 70f64), 1f64, (1.93, 2.7, 3.0), std::f64::consts::TAU/12f64);
}

//Obselete
fn detect_edges(image: &GrayImage) -> GrayImage {
    let mut edges = GrayImage::new(image.width(), image.height());
    for x in 0..image.width() {
        for y in 0..image.height() {
            let l = if x == 0 { 0 } else { image.get_pixel(x-1, y).0[0] as i32 };
            let u = if y == 0 { 0 } else { image.get_pixel(x, y-1).0[0] as i32 };
            let r = if x == (image.width()-1) { 0 } else { image.get_pixel(x+1, y).0[0] as i32 };
            let d = if y == (image.height()-1) { 0 } else { image.get_pixel(x, y+1).0[0] as i32 };

            let xgrad = -l + r;
            let ygrad = -u + d;

            edges.put_pixel(x, y, [max(xgrad.abs(), ygrad.abs()) as u8].into());
        }
    }
    edges
}

//Gaussian blur
fn blur(image: &GrayImage) -> GrayImage {
    let mut blurred = GrayImage::new(image.width(), image.height());
    for x in 1..image.width() - 1 {
        for y in 1..image.height() - 1 {
            let l = image.get_pixel(x-1, y).0[0] as u32;
            let lu = image.get_pixel(x-1, y-1).0[0] as u32;
            let u =  image.get_pixel(x, y-1).0[0] as u32;
            let ur = image.get_pixel(x+1, y-1).0[0] as u32;
            let r = image.get_pixel(x+1, y).0[0] as u32;
            let rd = image.get_pixel(x+1, y+1).0[0] as u32;
            let d =  image.get_pixel(x, y+1).0[0] as u32;
            let ld = image.get_pixel(x-1, y+1).0[0] as u32;
            let c = image.get_pixel(x, y).0[0] as u32;

            let new = (c*4 + r*2 + u*2 + l*2 + d*2 + lu + ur + rd + ld) as f64/16f64;

            blurred.put_pixel(x, y, [new as u8].into());
        }
    }
    blurred

}

///Harris corner detection. Can do edges and corners
fn detect_corners(image: &GrayImage) -> GrayImage {
    let mut corners = GrayImage::new(image.width(), image.height());
    let mut derivatives = vec![vec![(0f64, 0f64, 0f64); image.height() as usize]; image.width() as usize];
    let mut potential_corners = vec![vec![0f64; image.height() as usize]; image.width() as usize];
    //For each point on image, use the Sobel operator to calculate x and y spatial derivatives
    for x in 1..image.width() - 1 {
        let row = derivatives.get_mut(x as usize).unwrap();
        for y in 1..image.height() - 1 {
            let l = image.get_pixel(x-1, y).0[0] as f64;
            let lu = image.get_pixel(x-1, y-1).0[0] as f64;
            let u =  image.get_pixel(x, y-1).0[0] as f64;
            let ur = image.get_pixel(x+1, y-1).0[0] as f64;
            let r = image.get_pixel(x+1, y).0[0] as f64;
            let rd = image.get_pixel(x+1, y+1).0[0] as f64;
            let d =  image.get_pixel(x, y+1).0[0] as f64;
            let ld = image.get_pixel(x-1, y+1).0[0] as f64;

            let xsobel = lu + 2f64*l + ld - ur - 2f64*r - rd;
            let ysobel = lu + 2f64*u + ur - ld - 2f64*d - rd;
            //Store these 3 products, to save us calculating them later
            row[y as usize] = (xsobel*xsobel, ysobel*ysobel, xsobel*ysobel);
        }
    }
    //Add up the 3 products of note for the 3*3 region centered on every point
    for x in 1..image.width() as usize - 1 {
        for y in 1..image.height() as usize - 1 {
            let x_squared = derivatives[x-1][y-1].0+derivatives[x][y-1].0+derivatives[x+1][y-1].0+derivatives[x-1][y].0+derivatives[x][y].0+derivatives[x+1][y].0+derivatives[x-1][y+1].0+derivatives[x][y+1].0+derivatives[x+1][y+1].0;

            let y_squared = derivatives[x-1][y-1].1+derivatives[x][y-1].1+derivatives[x+1][y-1].1+derivatives[x-1][y].1+derivatives[x][y].1+derivatives[x+1][y].1+derivatives[x-1][y+1].1+derivatives[x][y+1].1+derivatives[x+1][y+1].1;

            let x_times_y = derivatives[x-1][y-1].2+derivatives[x][y-1].2+derivatives[x+1][y-1].2+derivatives[x-1][y].2+derivatives[x][y].2+derivatives[x+1][y].2+derivatives[x-1][y+1].2+derivatives[x][y+1].2+derivatives[x+1][y+1].2;

            //Calculate r value. If |r| = 0 (approx), neither corner nor edge. If r < 0, edge. If
            //r > 0, and r is big, corner.
            let matrix = [[x_squared, x_times_y], [x_times_y, y_squared]];
            let determinant = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0];
            let trace = matrix[0][0] + matrix[1][1];
            let r = determinant - HARRIS_CONST*trace*trace;

            if r > R_THRESHOLD {
                potential_corners[x][y] = r;
            }

        }
    }
    //Non-maximum suppression
    for x in MAX_WINDOW_SIZE.0..image.width() as usize - MAX_WINDOW_SIZE.1 {
        for y in MAX_WINDOW_SIZE.0..image.height() as usize - MAX_WINDOW_SIZE.1 {
            let mut r_values = Vec::with_capacity(MAX_WINDOW_SIZE.0*MAX_WINDOW_SIZE.1);
            for u in -(MAX_WINDOW_SIZE.0 as isize)..MAX_WINDOW_SIZE.1 as isize {
                for v in -(MAX_WINDOW_SIZE.0 as isize)..MAX_WINDOW_SIZE.1 as isize {
                    r_values.push(potential_corners[(x as i32 + u as i32) as usize][(y as i32 + v as i32) as usize]);
                }
            }
            let centre = potential_corners[x][y];
            let max = r_values.iter().map(|r| r <= &centre).fold(true, |acc, x| acc && x);
            if max && centre.abs() > 0.1 {
                corners.put_pixel(x as u32, y as u32, [255u8].into());
            }
        }
    }
    corners
}

//Finds the real-life 2d coordinates of a point on the screen, using a coordinate system centered on
//the camera, parallel to the ground, assuming the point is on the ground. Cam-orientation is given as 0 if the top of the viewport (y-coord=0) is pointing in the direction of the z-axis, and in radians going clockwise from the top of the viewport otherwise. Point is signed, from the centre of the viewport.
//This function is an abomination
fn real_coords(viewport_size_px: (f64, f64), viewport_size_metres: (f64, f64), point: (f64, f64), cam_height: f64, cam_to_viewport: (f64, f64, f64), cam_orientation: f64)-> (f64, f64) {
    let viewport_centre = cam_to_viewport;
    let cam_to_viewport_distance = (cam_to_viewport.0.powf(2f64)+cam_to_viewport.1.powf(2f64)+cam_to_viewport.2.powf(2f64)).powf(0.5f64);
    let cam_to_viewport_normalised = (cam_to_viewport.0/cam_to_viewport_distance, cam_to_viewport.1/cam_to_viewport_distance, cam_to_viewport.2/cam_to_viewport_distance);
    //The vector from the viewport to its intersection with the z axis will serve as our
    //y-direction. Normalised.
    let viewport_y_basis = (-cam_to_viewport_normalised.0, -cam_to_viewport_normalised.1, cam_to_viewport_normalised.2);
    //Take the cross product of the y_basis and cam_to_viewport to get the x direction. For later.
    let axis_cross_y_basis = (-2f64*cam_to_viewport_normalised.1*cam_to_viewport_normalised.2, 2f64*cam_to_viewport_normalised.0*cam_to_viewport_normalised.2, 0f64);
    //Rotate the y basis around the cam_to_viewport vector
    let rotated_viewport_y_basis = (cam_orientation.cos()*viewport_y_basis.0+cam_orientation.sin()*axis_cross_y_basis.0+(1f64-cam_orientation.cos())*(-cam_to_viewport_normalised.0.powf(2f64)-cam_to_viewport_normalised.1.powf(2f64)+cam_to_viewport_normalised.2.powf(2f64))*cam_to_viewport_normalised.0,
    cam_orientation.cos()*viewport_y_basis.1+cam_orientation.sin()*axis_cross_y_basis.1+(1f64-cam_orientation.cos())*(-cam_to_viewport_normalised.0.powf(2f64)-cam_to_viewport_normalised.1.powf(2f64)+cam_to_viewport_normalised.2.powf(2f64))*cam_to_viewport_normalised.1,
    cam_orientation.cos()*viewport_y_basis.2+cam_orientation.sin()*axis_cross_y_basis.2+(1f64-cam_orientation.cos())*(-cam_to_viewport_normalised.0.powf(2f64)-cam_to_viewport_normalised.1.powf(2f64)+cam_to_viewport_normalised.2.powf(2f64))*cam_to_viewport_normalised.2);
    //Take the cross product of the y basis and cam_to_viewport_normalised, to get the x basis
    let rotated_viewport_x_basis = (-cam_to_viewport_normalised.1*rotated_viewport_y_basis.2+cam_to_viewport_normalised.2*rotated_viewport_y_basis.1, -cam_to_viewport_normalised.2*rotated_viewport_y_basis.0+cam_to_viewport_normalised.0*rotated_viewport_y_basis.2, -cam_to_viewport_normalised.0*rotated_viewport_y_basis.1+cam_to_viewport_normalised.1*rotated_viewport_y_basis.0);
    //Converts a point in the viewport to a point in 3d space
    let point_coordinate = (rotated_viewport_x_basis.0*viewport_size_metres.0*point.0/viewport_size_px.0+rotated_viewport_y_basis.0*viewport_size_metres.1*point.1/viewport_size_px.1, rotated_viewport_x_basis.1*viewport_size_metres.0*point.0/viewport_size_px.0+rotated_viewport_y_basis.1*viewport_size_metres.1*point.1/viewport_size_px.1, rotated_viewport_x_basis.2*viewport_size_metres.0*point.0/viewport_size_px.0+rotated_viewport_y_basis.2*viewport_size_metres.1*point.1/viewport_size_px.1);
    //Ray from camera to point in 3d space. The feature at the point lies on this ray. As the
    //camera is at the origin, we just add the point_coordinate to the centre of the viewport.
    let ray = (point_coordinate.0+viewport_centre.0, point_coordinate.1+viewport_centre.1, point_coordinate.2+viewport_centre.2);
    println!("{:?}", ray);
    println!("{:?}", rotated_viewport_x_basis);
    println!("{:?}", rotated_viewport_y_basis);
    //Find length of ray at intersection of it and the ground
    let lambda = -cam_height/ray.2;
    //Gives x and y coordinates
    (ray.0*lambda, ray.1*lambda)
}

