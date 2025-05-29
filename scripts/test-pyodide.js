const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const { chromium } = require("playwright");
const http = require("http");
const serveStatic = require("serve-static");
const finalhandler = require("finalhandler");

const pyodideVersion = process.argv[2] || "0.27.5";

(async () => {
  console.log("🔧 Building wheel...");
  try {
    execSync("uv build", { stdio: "inherit" });
  } catch (e) {
    console.error("❌ uv build failed");
    process.exit(1);
  }

  // distディレクトリの絶対パス
  const distDir = path.resolve(__dirname, "../dist");
  const wheelFile = fs.readdirSync(distDir).find(f => f.endsWith(".whl"));
  if (!wheelFile) {
    console.error("❌ No .whl file found in dist/");
    process.exit(1);
  }

  console.log("📦 Found wheel:", wheelFile);

  // serve-staticでdistフォルダを配信
  const serve = serveStatic(distDir, { index: false });

  // HTTPサーバーを作成
  const server = http.createServer((req, res) => {
    // CORSヘッダーをつける
    res.setHeader("Access-Control-Allow-Origin", "*");
    serve(req, res, finalhandler(req, res));
  });

  // ポート8000でリスン開始
  const port = 8000;
  await new Promise((resolve) => server.listen(port, resolve));
  console.log(`🌐 HTTP server running at http://localhost:${port}/`);

  // ブラウザ起動
  const browser = await chromium.launch();
  const page = await browser.newPage();

  page.on("console", msg => console.log("> " + msg.text()));

  const blankHtmlPath = path.resolve(__dirname, "blank.html");
  const blankHtmlUrl = `file://${blankHtmlPath.replace(/\\/g, "/")}`;
  await page.goto(blankHtmlUrl);

  // wheelファイルのURLをHTTPサーバー経由で渡す
  const wheelUrl = `http://localhost:${port}/${wheelFile}`;

  await page.evaluate(
    async ({ wheelUrl, moduleName, pyodideVersion }) => {
      const pyodide = await import(`https://cdn.jsdelivr.net/pyodide/v${pyodideVersion}/full/pyodide.mjs`)
        .then(m => m.loadPyodide());

      await pyodide.loadPackage("micropip");

      try {
        await pyodide.runPythonAsync(`
          import micropip
          await micropip.install("${wheelUrl}")
          import ${moduleName}
          print("✅ ${moduleName} imported successfully")
        `);
      } catch (e) {
        console.error("❌ Pyodide install/import failed");
        throw e;
      }
    },
    { wheelUrl, moduleName: "easybench", pyodideVersion }
  );

  await browser.close();

  // HTTPサーバーを閉じる
  server.close();
})();
