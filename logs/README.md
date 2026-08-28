# 运行日志

`logs/` 统一保存仓库脚本产生的组合运行日志，例如训练、生成和评估共用的 `log_*.log`。

- 新脚本应把 `LOG` 写成 `${REPO_ROOT}/logs/<文件名>.log`。
- `outputs/<模型>/<实验>/training.log` 和 `sample.log` 仍留在各自实验目录，它们属于单次实验输出。
- 终端可以显示颜色，但写入文件的日志不得包含 ANSI 颜色控制码。
- 已停止实验的旧日志可按日期放入 `logs/archived/<日期>/`。

日志文件由 `.gitignore` 中的 `*.log` 规则忽略；本说明文件保留在 Git 中。
