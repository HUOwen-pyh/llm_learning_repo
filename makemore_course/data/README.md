# 数据

`names_mini.txt` 是用于快速测试的小型名字语料。

完整课程使用 Karpathy 的 `names.txt`，来源：

https://raw.githubusercontent.com/karpathy/makemore/master/names.txt

在项目根目录运行：

```powershell
python download_names.py
```

程序会把完整语料保存为 `data/names.txt`。不要把完整数据混入
`names_mini.txt`，否则快速检查会变慢且失去确定性。

