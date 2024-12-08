Build Docker image:
```shell
docker build -t transmet-compass .
```

Run Docker container:
```shell
docker run -it -v $(pwd):/workspace transmet-compass bash
```

In the container, the following command runs Compass with the input file `expression.tsv` from the `data` directory:
```shell
compass --data data/expression.tsv --species homo_sapiens --output-dir data
```
