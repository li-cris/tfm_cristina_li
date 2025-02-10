# Scripts

Regenerate [`ensembl_gene_info.tsv`](../resources/ensembl_gene_info.tsv):

```shell
python3 generate_ensembl_gene_info -o ../resources/ensembl_gene_info.tsv
```

Regenerate [`msigdb_go_gene_map.tsv`](../resources/msigdb_go_gene_map.tsv):

```shell
python3 generate_msigdb_go_gene_map.py -i ../resources/c5.go.bp.v2024.1.Hs.json.txt -e ../resources/ensembl_gene_info.tsv -o ../resources/msigdb_go_gene_map.tsv
```
