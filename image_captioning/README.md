安装 requirements.txt 中的包, 准备好环境。

下载资源:
```python
>>> import nltk
>>> nltk.download('punkt')
```

跑 baseline:
```bash
sbatch run.sh
```


注: 
evaluate.py 用于计算指标，预测结果 `prediction.json` 写成这样的形式:
```json
[
    {
        "img_id":"1386964743_9e80d96b05.jpg",
        "prediction":[
            "young boy is standing in a field of grass"
        ]
    },
    {
        "img_id":"3523559027_a65619a34b.jpg",
        "prediction":[
            "young boy is standing in a field of grass"
        ]
    },
    ......
]
```
调用方法：
```bash
python evaluate.py --prediction_file prediction.json \
                   --reference_file /lustre/home/acct-stu/stu168/data/image_captioning/flickr8k/caption.txt \
                   --output_file result.txt
```

