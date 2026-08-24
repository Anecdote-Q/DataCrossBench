# 匿名发布步骤

`anonymous.4open.science` 当前的工作方式是从 GitHub 仓库导入并生成匿名副本，不是直接上传本地目录。因此发布需要两个步骤。

## 1. 准备 GitHub 源仓库

本目录已经是可提交内容。因为数据中存在超过 100 MB 的文件，需要 Git LFS：

```bash
git lfs install
git clone https://github.com/<your-account>/<temporary-repo>.git
rsync -a --delete /path/to/DataCrossBench-anonymous/ /path/to/temporary-repo/
cd /path/to/temporary-repo
git add .
git commit -m "Release DataCrossBench benchmark and evaluator"
git push origin main
```

不要把 GitHub token 写入远程 URL、README 或脚本。推荐使用 GitHub CLI 登录、系统凭据管理器，或在本机环境变量中配置临时 token。

## 2. 导入到 4open

登录 `https://anonymous.4open.science/`，选择从 GitHub repository anonymize/import，填写上一步的公开仓库 URL。完成后会得到类似：

```text
https://anonymous.4open.science/r/<anonymous-id>/
```

4open 的登录会话不能由本地文件或 GitHub token 自动替代；需要在网页中完成一次 GitHub OAuth/站点登录。当前执行环境没有可用浏览器会话，因此本次只能准备好本地发布副本，不能替用户完成最后的网页操作。

## 凭据说明

- 不需要把 token 粘贴到对话中。
- 若由自动化环境推送 GitHub，需要一个具备创建/写入目标仓库权限的临时 GitHub token，以及本机已安装 Git LFS。
- 若 GitHub 仓库已经由你创建并推送完毕，只需要在 4open 网页登录并提交仓库 URL。
- 公开前请确认数据、图像和第三方来源允许再分发。
