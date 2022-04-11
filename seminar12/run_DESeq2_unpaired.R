library(DESeq2)

counts <- read.table("colon_cancer_tumor_vs_normal_unpaired_counts.tsv", sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
groups <- read.table("colon_cancer_tumor_vs_normal_unpaired_annotation.tsv", sep = "\t", header = TRUE)
expr_group <- "Tumor"
ctrl_group <- "Normal"

groups$Group <- relevel(factor(groups$Group), ref = ctrl_group)
rownames(groups) <- groups$Sample
groups$Sample <- NULL

dds <- DESeqDataSetFromMatrix(countData = counts,
                              colData = groups,
                              design = ~ Group)

dds <- DESeq(dds)
res <- lfcShrink(dds, coef = paste("Group_", expr_group, "_vs_", ctrl_group, sep=""), type = "apeglm")
res <- res[order(res$padj),]

write.table(res, file = "results.tsv", sep = "\t", quote = FALSE)
