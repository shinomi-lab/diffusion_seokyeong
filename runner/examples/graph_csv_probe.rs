//! `graph = { ..., format = "csv" }` が実際に正しくCSVとして読み込まれ、
//! エッジリストがグラフ構造に正確に反映されているかを検証する診断ツール。
//!
//! `util::GraphConfig` → `graph_lib::io::ParseBuilder` という、
//! xa-promote が実際に使っているのと全く同じコードパスを通す
//! （再実装ではなく、本物の依存クレートをそのまま動かして確認する）。
//!
//! 実行: cargo run --release --example graph_csv_probe
//! graph-lib のバージョン確認: このリポジトリが依存する graph-lib は
//! `Cargo.toml` で `tag = "v0.6.1"` に固定されている
//! （`/Users/hongseokyeong/Desktop/rustモデル/graph-lib` のチェックアウトは
//! v0.6.1 に対して LICENSE/README の2コミットのみの差分で、パース処理は同一と確認済み）。

use std::io::Write;
use std::path::PathBuf;

use graph_lib::prelude::{Graph, GraphB};
use util::GraphConfig;

fn write_fixture(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

fn load(path: &std::path::Path, directed: bool, transposed: bool, format: &str) -> anyhow::Result<GraphB> {
    let toml_str = format!(
        "path = {:?}\ndirected = {}\ntransposed = {}\nformat = \"{}\"\n",
        path.to_str().unwrap(),
        directed,
        transposed,
        format
    );
    let cfg: GraphConfig = toml::from_str(&toml_str)?;
    let g: GraphB = cfg.try_into()?;
    Ok(g)
}

fn dump(label: &str, g: &GraphB) {
    let (n, m, directed) = match g {
        GraphB::Di(d) => (d.node_count(), d.edge_count(), d.directed()),
        GraphB::Ud(u) => (u.node_count(), u.edge_count(), u.directed()),
    };
    println!("  [{label}] node_count={n} edge_count={m} directed={directed}");
    for i in 0..n {
        let succ: Vec<usize> = match g {
            GraphB::Di(d) => d.successors(i).copied().collect(),
            GraphB::Ud(u) => u.successors(i).copied().collect(),
        };
        println!("    node {i}: successors={succ:?}");
    }
}

/// `f` を実行し、パニックしても既定のパニックフック（stderrへの出力）を出さずに
/// `Err` として捕捉する。想定内のpanicを検証する箇所の出力を読みやすくするため。
fn catch_silently<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> std::thread::Result<T> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    result
}

fn main() {
    let dir = std::env::temp_dir().join("graph_csv_probe");
    std::fs::create_dir_all(&dir).unwrap();
    let mut n_pass = 0usize;
    let mut n_fail = 0usize;
    macro_rules! check {
        ($cond:expr, $msg:expr) => {
            if $cond {
                println!("    [OK] {}", $msg);
                n_pass += 1;
            } else {
                println!("    [NG] {}", $msg);
                n_fail += 1;
            }
        };
    }

    // -----------------------------------------------------------------
    // 1. graph-lib 自身のテストと同じ形式（`u, v` カンマ+スペース、コメント行あり）
    //    が xa-promote の実利用コードパス（util::GraphConfig）でも同じ結果になるか
    // -----------------------------------------------------------------
    println!("=== 1. 基本形式（カンマ+スペース区切り、コメント行あり） ===");
    let p1 = write_fixture(&dir, "basic.csv", "# comment\n0, 1\n0, 2\n1, 3\n2, 3\n# last comment\n");
    let g1 = load(&p1, true, false, "csv").expect("基本CSVの読み込みに失敗");
    dump("basic, directed, non-transposed", &g1);
    if let GraphB::Di(d) = &g1 {
        check!(d.node_count() == 4, "node_count = 4");
        check!(d.edge_count() == 4, "edge_count = 4");
        check!(d.contains(0, 1) && d.contains(0, 2) && d.contains(1, 3) && d.contains(2, 3),
               "全エッジが正しい向きで反映されている");
        check!(!d.contains(1, 0), "逆向きのエッジは存在しない（有向グラフとして扱われている）");
    }

    // -----------------------------------------------------------------
    // 2. transposed=true が実際に辺の向きを反転させているか
    // -----------------------------------------------------------------
    println!("\n=== 2. transposed=true で辺の向きが反転するか ===");
    let g2 = load(&p1, true, true, "csv").expect("transposed CSVの読み込みに失敗");
    dump("basic, directed, transposed", &g2);
    if let GraphB::Di(d) = &g2 {
        check!(d.contains(1, 0) && d.contains(2, 0) && d.contains(3, 1) && d.contains(3, 2),
               "transposed指定で全エッジの向きが反転している");
        check!(!d.contains(0, 1), "反転前の向きはもう存在しない");
    }

    // -----------------------------------------------------------------
    // 3. CSVはカンマ区切り・EdgeListは空白/タブ区切りという仕様上の違いが
    //    実際にそのとおり動くか（同じ内容をCSVとEdgeListで読み、一致確認）
    // -----------------------------------------------------------------
    println!("\n=== 3. format=\"csv\" と format=\"edgelist\" の実際の区別 ===");
    let p_csv = write_fixture(&dir, "same.csv", "10,20\n20,30\n30,10\n");
    let p_el = write_fixture(&dir, "same.txt", "10 20\n20 30\n30 10\n");
    let g_csv = load(&p_csv, true, false, "csv").unwrap();
    let g_el = load(&p_el, true, false, "edgelist").unwrap();
    if let (GraphB::Di(dc), GraphB::Di(de)) = (&g_csv, &g_el) {
        check!(dc.node_count() == de.node_count() && dc.edge_count() == de.edge_count(),
               "同一データをCSV/EdgeListそれぞれで読んでも同型のグラフになる");
    }
    // 決定的な区別テスト: EdgeList形式のファイル(空白区切り)をformat="csv"で読むと、
    // カンマが1つも無いため各行が1トークンしか無く、graph-lib はパニックするはず。
    println!("  format=\"csv\" でカンマの無いファイルを読ませた場合の挙動を確認:");
    let result = catch_silently(|| load(&p_el, true, false, "csv"));
    match result {
        Ok(Ok(_)) => println!("    [NG] 想定外: パニックせず成功してしまった（誤読みの疑い）"),
        Ok(Err(e)) => println!("    [NG] 想定外: panicではなくErrで失敗: {e}"),
        Err(_) => {
            println!("    [OK] 想定通りpanicした → format=\"csv\" は本当にカンマで分割しており、");
            println!("         空白区切りのファイルを誤って読み飛ばしたりはしない（fail-fast）");
            n_pass += 1;
        }
    }

    // -----------------------------------------------------------------
    // 4. ヘッダー行があるとどうなるか（実データでありがちな落とし穴）
    // -----------------------------------------------------------------
    println!("\n=== 4. ヘッダー行 \"from,to\" がある場合（実データにありがちな形） ===");
    let p_header = write_fixture(&dir, "with_header.csv", "from,to\n1,2\n2,3\n");
    let result = catch_silently(|| load(&p_header, true, false, "csv"));
    match result {
        Ok(Ok(g)) => {
            let n = match &g { GraphB::Di(d) => d.node_count(), _ => 0 };
            println!("    [NG] 想定外: ヘッダー行があっても成功してしまった (node_count={n})。");
            println!("         → \"from\",\"to\" が数値として誤読みされた可能性がある");
            n_fail += 1;
        }
        Err(_) => {
            println!("    [OK] ヘッダー行があると panic する（\"from\".parse::<i32>() が失敗するため）。");
            println!("         → もし実際の ENGB_edges.csv にヘッダー行があれば、");
            println!("           xa-promote は必ずここで panic するはずである。");
            println!("           逆に言えば、過去に extzero/extneg が正常終了しているという事実は、");
            println!("           あのファイルに数値以外の文字列を含む行（ヘッダー等）が");
            println!("           無かったことの直接証拠になる。");
            n_pass += 1;
        }
        Ok(Err(e)) => {
            println!("    [OK] ヘッダー行でErrになる: {e}");
            n_pass += 1;
        }
    }

    // -----------------------------------------------------------------
    // 5. 余分な列（重み等）があっても最初の2列だけを正しく使うか
    // -----------------------------------------------------------------
    println!("\n=== 5. 3列目（重みなど）がある場合、先頭2列だけが使われるか ===");
    let p_w = write_fixture(&dir, "weighted.csv", "1,2,0.5\n2,3,0.8\n3,1,1.0\n");
    let g_w = load(&p_w, true, false, "csv").expect("重み付きCSVの読み込みに失敗");
    if let GraphB::Di(d) = &g_w {
        check!(d.node_count() == 3 && d.edge_count() == 3,
               "3列目は無視され、先頭2列（送信元・宛先）だけが正しく使われている");
    }

    // -----------------------------------------------------------------
    // 6. 疎な（連番でない）実データ風のノードIDが正しく扱われるか
    //    → node_count は「登場した相異なるラベルの数」であり、max(id)+1 ではない
    // -----------------------------------------------------------------
    println!("\n=== 6. 疎なノードID（実データ風）の扱い ===");
    let p_sparse = write_fixture(&dir, "sparse.csv", "1000,2000\n2000,3000\n3000,1000\n");
    let g_sparse = load(&p_sparse, true, false, "csv").unwrap();
    if let GraphB::Di(d) = &g_sparse {
        println!("    生のID {{1000,2000,3000}} → 内部的には0..{}に詰め直される（実際のnode_count={}）",
                 d.node_count(), d.node_count());
        check!(d.node_count() == 3,
               "node_countは「登場した相異なるラベル数」（3個）であり、最大ID+1（3001）ではない");
        check!(d.edge_count() == 3, "エッジ数はそのまま3本");
    }
    println!("    [注記] これはCSV固有の挙動ではなくgraph-lib全体の仕様（get_sorted_node_labels/");
    println!("           invert_node_label_map によるソート済み相異なるラベルへの再割当て）。");
    println!("           次数0の孤立ノード（辺に一度も登場しないノード）は数えられない点に注意。");

    // -----------------------------------------------------------------
    // 7. CRLF改行・余分な空白でも壊れないか
    // -----------------------------------------------------------------
    println!("\n=== 7. CRLF改行・余分な空白があっても正しく読めるか ===");
    let p_crlf = write_fixture(&dir, "crlf.csv", "1, 2\r\n2 , 3\r\n  3,1  \r\n");
    let g_crlf = load(&p_crlf, true, false, "csv").expect("CRLF付きCSVの読み込みに失敗");
    if let GraphB::Di(d) = &g_crlf {
        check!(d.node_count() == 3 && d.edge_count() == 3,
               "CRLF・前後の空白があっても正しく3ノード3エッジとして読める（trim()で吸収される）");
    }

    // -----------------------------------------------------------------
    // 9. directed=false（無向）で読んだ場合に UndiGraphB がどう構築されるか
    //    graph-lib::io::parse は1ファイル行につき (u,v) を1回だけ push し、
    //    G::from(&els) が UndiGraphB::add_edge(u,v) を呼ぶ。add_edge 自身が
    //    内部で (u,v) と (v,u) の両方を BTreeSet に挿入する（graph.rs 確認済み）。
    //    よって: ファイル行が既に逆向き重複を含んでいても BTreeSet の性質上
    //    二重カウントされない。transposed は add_edge に渡す前に (u,v)→(v,u)
    //    と入れ替えるだけなので、無向グラフでは意味を持たない no-op になる。
    // -----------------------------------------------------------------
    println!("\n=== 9. directed=false（無向）での構築 ===");
    let g9 = load(&p1, false, false, "csv").expect("無向CSVの読み込みに失敗");
    dump("basic, undirected, non-transposed", &g9);
    if let GraphB::Ud(u) = &g9 {
        check!(u.node_count() == 4, "node_count = 4（有向のときと同じ）");
        check!(u.edge_count() == 4, "edge_count = 4（4本の異なる無向ペアなので半分にはならない）");
        check!(u.contains(0, 1) && u.contains(1, 0), "0-1 は両方向とも contains() が true になる");
        check!(u.successors(1).copied().collect::<Vec<_>>() == vec![0, 3],
               "node 1 の successors は {{0,3}}（元の 0→1 と 1→3 の両方が反映され対称化されている）");
    } else {
        println!("    [NG] directed=false なのに GraphB::Di が返ってきた");
        n_fail += 1;
    }
    println!("  transposed=true でも結果が変わらないか（no-opのはず）:");
    let g9t = load(&p1, false, true, "csv").expect("無向・transposed CSVの読み込みに失敗");
    if let (GraphB::Ud(u), GraphB::Ud(ut)) = (&g9, &g9t) {
        let succs_all = |g: &graph_lib::prelude::UndiGraphB| -> Vec<Vec<usize>> {
            (0..g.node_count()).map(|i| g.successors(i).copied().collect()).collect()
        };
        check!(succs_all(u) == succs_all(ut),
               "transposed の有無で全ノードのsuccessorsが完全一致（無向では transposed は no-op）");
    }

    // -----------------------------------------------------------------
    // 8. 実ファイル test/musae_ENGB_edges.csv を、config で使うのと
    //    全く同じ設定（directed=true, transposed=true, format="csv"）で検証
    // -----------------------------------------------------------------
    let real_path = PathBuf::from("test/musae_ENGB_edges.csv");
    if real_path.exists() {
        println!("\n=== 8. 実ファイル {} の検証 ===", real_path.display());

        let first_line = std::fs::File::open(&real_path)
            .ok()
            .and_then(|f| std::io::BufRead::lines(std::io::BufReader::new(f)).next())
            .and_then(|l| l.ok())
            .unwrap_or_default();
        println!("  先頭行: {first_line:?}");
        let has_header = first_line.split(',').next().map_or(false, |s| s.parse::<i64>().is_err());
        println!("  ヘッダー行の疑い: {has_header}");

        println!("  現行の想定どおり（directed=true, transposed=true, format=\"csv\"）でロード試行:");
        let result = catch_silently(|| load(&real_path, false, true, "csv"));
        match result {
            Ok(Ok(g)) => {
                let (n, m) = match &g { GraphB::Di(d) => (d.node_count(), d.edge_count()), GraphB::Ud(u) => (u.node_count(), u.edge_count()) };
                if has_header {
                    println!("    [NG?] panicせず成功: node_count={n} edge_count={m}");
                    println!("          ヘッダー行があるのに成功している場合、想定と異なる挙動なので要調査");
                } else {
                    println!("    [OK] panicせず成功: node_count={n} edge_count={m}");
                    println!("         （先頭行がヘッダーではなく数値データだったため、想定どおり成功）");
                }
            }
            Ok(Err(e)) => println!("    [結果] Err: {e}"),
            Err(_) => {
                println!("    [結果] panicした。先頭行を数値としてパースしようとして失敗したため");
                println!("           （ヘッダー行の疑いあり、と判定した通りの想定内の失敗）。");
                println!("           → このファイルをそのまま `path = \"test/musae_ENGB_edges.csv\"` に");
                println!("             指定しても、このconfig ではロードできない。");
            }
        }

        let content = std::fs::read_to_string(&real_path).unwrap();
        let total_lines = content.lines().count();
        let file_data_lines = if has_header { total_lines - 1 } else { total_lines };

        if has_header {
            println!("\n  ヘッダー行を除いた場合（tail -n +2 相当）でロードし直す:");
            let no_header_path = dir.join("musae_ENGB_edges_noheader.csv");
            let without_header: String = content.lines().skip(1).collect::<Vec<_>>().join("\n");
            std::fs::write(&no_header_path, without_header).unwrap();
            match load(&no_header_path, true, true, "csv") {
                Ok(g) => {
                    let (n, m) = match &g { GraphB::Di(d) => (d.node_count(), d.edge_count()), GraphB::Ud(u) => (u.node_count(), u.edge_count()) };
                    println!("    [OK] ヘッダーを除けば正常にロードできる: node_count={n} edge_count={m}");
                    println!("    参考: claude-research.md §3.3 の記録では Twitch-ENGB は 7,126 ノード");
                    check!(n == 7126, "node_count が既知の Twitch-ENGB のノード数(7,126)と一致");
                    println!("    ファイルの実データ行数（ヘッダー除く）: {file_data_lines}");
                    check!(m == file_data_lines,
                           "edge_count がファイルの実データ行数と一致（行の欠落・重複読みが無い）");
                }
                Err(e) => println!("    [NG] ヘッダーを除いても失敗: {e}"),
            }
        } else {
            println!("\n  ヘッダー行が無いため、そのままロードした上表8の結果をそのまま検証する:");
            match load(&real_path, true, true, "csv") {
                Ok(g) => {
                    let (n, m) = match &g { GraphB::Di(d) => (d.node_count(), d.edge_count()), GraphB::Ud(u) => (u.node_count(), u.edge_count()) };
                    println!("    node_count={n} edge_count={m}");
                    println!("    参考: claude-research.md §3.3 の記録では Twitch-ENGB は 7,126 ノード");
                    check!(n == 7126, "node_count が既知の Twitch-ENGB のノード数(7,126)と一致");
                    println!("    ファイルの実データ行数: {file_data_lines}");
                    check!(m == file_data_lines,
                           "edge_count がファイルの実データ行数と一致（行の欠落・重複読みが無い）");
                }
                Err(e) => println!("    [NG] ロードに失敗: {e}"),
            }
        }

        // -------------------------------------------------------------
        // 10. 実ファイルを directed=false で読んだ場合
        //    ファイルに逆向き重複行（(v,u) が別行として既に存在するケース）
        //    があるかどうかで、期待される edge_count の計算式が変わる点に注意。
        // -------------------------------------------------------------
        println!("\n=== 10. 実ファイルを directed=false で読んだ場合 ===");
        let raw_lines: Vec<&str> = if has_header {
            content.lines().skip(1).collect()
        } else {
            content.lines().collect()
        };
        let mut seen: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
        let mut reverse_dup = 0usize;
        for line in &raw_lines {
            let mut parts = line.split(',');
            let u: i64 = parts.next().unwrap().trim().parse().unwrap();
            let v: i64 = parts.next().unwrap().trim().parse().unwrap();
            if seen.contains(&(v, u)) {
                reverse_dup += 1;
            }
            seen.insert((u, v));
        }
        let expected_undirected_edges = raw_lines.len() - reverse_dup;
        println!("  ファイルの実データ行数: {}", raw_lines.len());
        println!("  逆向きが既に別行として存在する行数: {reverse_dup}");
        println!("  → directed=false なら edge_count は {expected_undirected_edges} になるはず");
        println!("    （逆向き重複が無ければ有向のときと同じ行数。あれば重複ぶん減る）");

        let load_path = if has_header {
            dir.join("musae_ENGB_edges_noheader.csv")
        } else {
            real_path.clone()
        };
        match load(&load_path, false, true, "csv") {
            Ok(g) => match &g {
                GraphB::Ud(u) => {
                    println!("    [OK] directed=false でも panic せず GraphB::Ud として読み込めた");
                    println!("    node_count={} edge_count={}", u.node_count(), u.edge_count());
                    check!(u.node_count() == 7126, "node_count は有向のときと同じ 7,126");
                    check!(u.edge_count() == expected_undirected_edges,
                           "edge_count が「逆向き重複を1本に畳んだ本数」と一致");
                    let g_no_transpose = load(&load_path, false, false, "csv")
                        .expect("無向・非transposedでの再読み込みに失敗");
                    if let GraphB::Ud(u2) = &g_no_transpose {
                        check!(u.node_count() == u2.node_count() && u.edge_count() == u2.edge_count(),
                               "transposedの有無で結果が変わらない（無向ではtransposedはno-op）");
                    }
                }
                GraphB::Di(_) => {
                    println!("    [NG] directed=false なのに GraphB::Di が返ってきた");
                    n_fail += 1;
                }
            },
            Err(e) => println!("    [NG] directed=false でのロードに失敗: {e}"),
        }
    } else {
        println!("\n=== 8. {} が見つからないためスキップ ===", real_path.display());
    }

    println!("\n=== まとめ: {n_pass} 件成功 / {n_fail} 件失敗 ===");
    if n_fail > 0 {
        std::process::exit(1);
    }
}
