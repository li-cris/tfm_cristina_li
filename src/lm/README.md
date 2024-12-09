Build Docker image:
```shell
docker build -t transmet-lm .
```

Run Docker container:
```shell
docker run -it -v $(pwd):/workspace transmet-lm bash
```
