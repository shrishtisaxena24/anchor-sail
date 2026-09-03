# Anchor & Sail dashboard — one-time setup (about 15 minutes, no coding)

After this setup the dashboard rebuilds itself on GitHub's servers every trading hour and
every evening. You never run anything again — you just open the link.

> **Why GitHub?** A browser page cannot fetch prices on its own (Dhan and Yahoo both block
> browser calls). GitHub Actions is a free scheduler that runs the Python engine and publishes
> the result; GitHub Pages hosts the page at a permanent URL.

---

## Step 1 — Create a GitHub account (skip if you have one)

1. Go to <https://github.com/signup> and create an account (free plan).
2. Verify the email address GitHub sends you.

## Step 2 — Create the repository

1. Click the **+** at the top-right → **New repository**.
2. Repository name: `anchor-sail`
3. Choose **Public** (GitHub Pages is free only on public repositories; the URL is not listed
   anywhere, but anyone who has the link can open it. If you want it private, GitHub Pro at
   about $4/month allows Pages on private repositories.)
4. Tick **Add a README file**.
5. Click **Create repository**.

## Step 3 — Upload the dashboard files

1. Unzip `anchor-sail.zip` on your computer.
2. In the repository page click **Add file** → **Upload files**.
3. Drag these folders and files from the unzipped folder into the upload area:
   `engine`, `data`, `docs`, `tests`, `requirements.txt`, `README.md`, `SETUP.md`
   (drag the folders themselves — GitHub keeps the folder structure).
4. Scroll down, click **Commit changes**.

## Step 4 — Add the scheduler file (the one hidden folder)

Finder hides folders that start with a dot, so this one is created by paste:

1. Click **Add file** → **Create new file**.
2. In the file-name box type exactly: `.github/workflows/daily_dashboard.yml`
   (typing the `/` automatically creates the folders).
3. Open `daily_dashboard.yml` from the unzipped folder (inside `.github/workflows/`) in
   TextEdit, select all, copy, and paste into the big editor box on GitHub.
4. Click **Commit changes**.

## Step 5 — Allow the scheduler to save data

1. **Settings** (tab at the top of the repository) → left menu **Actions** → **General**.
2. Scroll to **Workflow permissions** → choose **Read and write permissions** → **Save**.

## Step 6 — Turn on the web page

1. **Settings** → left menu **Pages**.
2. Under **Build and deployment** → **Source**: *Deploy from a branch*.
3. **Branch**: `main`, folder: **/docs** → **Save**.
4. After a minute the page shows your URL:
   `https://<your-username>.github.io/anchor-sail/` — bookmark it (also works on your phone).

## Step 7 — First run

1. Click the **Actions** tab → **Anchor & Sail daily dashboard** (left) → **Run workflow** →
   green **Run workflow** button.
2. Wait 4–8 minutes (first run downloads ~10 years of prices for ~500 stocks).
   A green tick means success; click the run to see the log summary.
3. Open your Pages URL. If it still says "No data yet", wait one more minute (Pages
   republishes after each data update) and reload.

From now on it runs automatically: hourly 10:15–15:15 IST on trading days, at 15:45 IST,
and a final pass at 18:00 IST every day. Opening the URL always shows the latest build;
the page also refreshes itself every 5 minutes while open.

---

## Optional: official Midcap 150 / Smallcap 250 benchmark files

Yahoo carries `NIFTYMIDCAP150.NS` and `NIFTYSMLCAP250.NS`; if either disappears the engine
falls back to a labelled ETF proxy and shows a warning. To force the official series, download
the historical CSV from niftyindices.com (Reports → Historical Data → index → date range) and
upload it as `data/benchmarks/PRECISION.csv` or `data/benchmarks/FRONTIER.csv`
(columns `Date` and `Close`). The engine uses the CSV whenever it is present.

## If a run fails

Actions → click the red run → read the summary. The most common cause is Yahoo being slow;
the next scheduled run simply retries and the dashboard keeps showing the last good data.
Nothing in the ledger is lost — the open book lives in `data/state/*.json` and is committed
after every successful run.

## Changing the strategy parameters

Do not — the brief says the strategy must not change. Everything is in `engine/strategy.py`
(parameters at the top) and `engine/build.py` (`INCEPTION_MONTH`, portfolio table). If a
change is ever agreed, edit the file on GitHub (pencil icon) and the next run picks it up.
