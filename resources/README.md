# Resources

## `c5.go.bp.v2024.1.Hs.json.txt`

This file was downloaded from collection [C5](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#C5) of [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/index.jsp) [^1].

We use the first subcollection, derived from the Gene Ontology (GO) [^2] resource which contains Biological Process (BP), Cellular Component (CC), and Molecular Function (MF) components.

We downloaded the [JSON bundle of the BP component](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2024.1.Hs/c5.go.bp.v2024.1.Hs.json), comprising 7608 gene sets derived from the GO BP ontology.

## `ensembl_gene_info.tsv`

This file contains information for all protein-coding genes in the human genome from Ensembl.
It can be regenerated using the script [`generate_ensembl_gene_info.py`](../scripts/generate_ensembl_gene_info.py).

## `msigdb_go_gene_map.tsv`

This file contains the GO pathway-gene map from MSigDB, as given by the file [`c5.go.bp.v2024.1.Hs.json.txt`](c5.go.bp.v2024.1.Hs.json.txt).
It can be regenerated using the script [`generate_msigdb_go_gene_map.py`](../scripts/generate_msigdb_go_gene_map.py).

## `sena_go_gene_map.tsv`

This file contains the GO pathway-gene map from the file [`go_kegg_gene_map.tsv`](https://raw.githubusercontent.com/ML4BM-Lab/SENA/refs/heads/master/data/go_kegg_gene_map.tsv) from the [SENA repository](https://github.com/ML4BM-Lab/SENA).

## `sena_kegg_gene_map.tsv`

This file contains the KEGG [^2] pathway-gene map from the file [`go_kegg_gene_map.tsv`](https://raw.githubusercontent.com/ML4BM-Lab/SENA/refs/heads/master/data/go_kegg_gene_map.tsv) from the [SENA repository](https://github.com/ML4BM-Lab/SENA).

## References

[^1]: [Subramanian et al. (2005)](https://doi.org/10.1073/pnas.0506580102)

[^2]: [Ashburner et al. (2000)](https://doi.org/10.1038/75556)

[^3]: [Kanehisa and Goto (2000)](https://doi.org/10.1093/nar/28.1.27)
