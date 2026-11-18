# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arukh89/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arukh89/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, as five quantiles an hour. `main.py` (Python 3.13, standard library only) reads one case from `/input`, fits two scale-conditioned models to the case's own comparable launches, and writes the forecast to `/output`. Both models work in log-ratios to each launch's hour-1 print: one fits a weighted least-squares line per hour over the 18 launches closest in scale, counting launches released on the same weekday four times as heavily; the other fits a weighted line over all launches, weighting each by how rolling-midnight-like its own day was (launches that peaked near hour 1 keep full weight; launches that exploded to many times hour 1 fade out). The two forecasts are averaged, with residual-based quantile spreads, and the curves are lightly smoothed across hours. Leave-one-out pinball loss over the kit's 37 launches was 0.193 on the 24 rolling-midnight-like launches (persistence baseline 0.265) and 0.247 on launches opening at 10k+ viewers. Deterministic, offline, well inside the 8-second cap.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards
