use crate::util;
use std::collections::{BTreeMap, BTreeSet};
use std::iter;
type ScalarFn = fn(f64) -> f64;
type DiffFn = fn(f64, f64, ScalarFn) -> f64;

fn forward_diff(x: f64, h: f64, f: ScalarFn) -> f64 {
    (f(x + h) - f(x)) / h
}

fn backward_diff(x: f64, h: f64, f: ScalarFn) -> f64 {
    (f(x) - f(x - h)) / h
}

fn center_diff(x: f64, h: f64, f: ScalarFn) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}
fn special_diff(x: f64, h: f64, f: ScalarFn) -> f64 {
    2.0 * forward_diff(x, h, f) - forward_diff(x, 2.0 * h, f)
}
fn observed_order(err_prev: f64, err_curr: f64) -> Option<f64> {
    let a = err_prev.abs();
    let b = err_curr.abs();

    if a == 0.0 || b == 0.0 {
        None
    } else {
        Some((a / b).ln() / 2.0_f64.ln())
    }
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    f: ScalarFn,
    df: ScalarFn,
}

#[derive(Clone, Copy)]
struct Method {
    name: &'static str,
    d: DiffFn,
}

#[derive(Debug, Clone)]
pub struct Row {
    pub case: &'static str,
    pub method: &'static str,
    pub h: f64,
    pub approx: f64,
    pub err: f64,
    pub order: Option<f64>,
}

fn eval_method(x: f64, hs: &[f64], case: Case, method: Method) -> Vec<Row> {
    let exact = (case.df)(x);

    hs.iter()
        .copied()
        .map(|h| {
            let approx = (method.d)(x, h, case.f);
            let err = exact - approx;
            (h, approx, err)
        })
        .scan(None, |prev_err: &mut Option<f64>, (h, approx, err)| {
            let order = prev_err.and_then(|e_prev| observed_order(e_prev, err));
            *prev_err = Some(err);

            Some(Row {
                case: case.name,
                method: method.name,
                h,
                approx,
                err,
                order,
            })
        })
        .collect()
}

pub fn compare(input: f64) -> Vec<Row> {
    let cases = [
        Case {
            name: "sqrt(x+1)",
            f: |x| (x + 1.0).sqrt(),
            df: |x| 0.5 / (x + 1.0).sqrt(),
        },
        Case {
            name: "exp(x)",
            f: f64::exp,
            df: f64::exp,
        },
    ];

    let methods = [
        Method {
            name: "forward",
            d: forward_diff,
        },
        Method {
            name: "backward",
            d: backward_diff,
        },
        Method {
            name: "center",
            d: center_diff,
        },
        Method {
            name: "special",
            d: special_diff,
        },
    ];

    let hs: Vec<f64> = iter::successors(Some(0.5_f64), |h| Some(h / 2.0))
        .take(12)
        .collect();

    let hs_ref = &hs;

    cases
        .into_iter()
        .flat_map(|case| {
            methods
                .into_iter()
                .flat_map(move |method| eval_method(input, hs_ref, case, method))
        })
        .collect()
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    let mut prev_us = false;

    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_us = false;
        } else if !prev_us {
            out.push('_');
            prev_us = true;
        }
    }

    out.trim_matches('_').to_string()
}

pub fn write_compare_npy(rows: &[Row]) {
    let out_dir = "data/ch2_2".to_string();

    let mut groups: BTreeMap<(&'static str, &'static str), Vec<&Row>> = BTreeMap::new();
    for r in rows {
        groups.entry((r.case, r.method)).or_default().push(r);
    }

    for rs in groups.values_mut() {
        rs.sort_by(|a, b| b.h.partial_cmp(&a.h).unwrap());
    }

    let mut wrote_exact_for_case: BTreeSet<&'static str> = BTreeSet::new();

    for ((case, method), rs) in groups {
        let case_slug = slug(case);
        let method_slug = slug(method);

        let hs: Vec<f64> = rs.iter().map(|r| r.h).collect();
        let approx: Vec<f64> = rs.iter().map(|r| r.approx).collect();
        let err: Vec<f64> = rs.iter().map(|r| r.err).collect();
        let abs_err: Vec<f64> = rs.iter().map(|r| r.err.abs()).collect();
        let order: Vec<f64> = rs.iter().map(|r| r.order.unwrap_or(f64::NAN)).collect();

        let base = format!("{}__{}", case_slug, method_slug);

        util::write_data(&hs, out_dir.clone(), format!("{}__h", base));
        util::write_data(&approx, out_dir.clone(), format!("{}__approx", base));
        util::write_data(&err, out_dir.clone(), format!("{}__err", base));
        util::write_data(&abs_err, out_dir.clone(), format!("{}__abs_err", base));
        util::write_data(&order, out_dir.clone(), format!("{}__order", base));

        if !wrote_exact_for_case.contains(case) && !rs.is_empty() {
            let exact = rs[0].approx + rs[0].err;
            util::write_data(&[exact], out_dir.clone(), format!("{}__exact", case_slug));
            wrote_exact_for_case.insert(case);
        }
    }
}
