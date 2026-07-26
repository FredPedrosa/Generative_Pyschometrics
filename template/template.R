# ==============================================================================
# TEMPLATE COMPLETO: PSEUDO-CFA COM EMBEDDINGS 
# ==============================================================================

# 1. CARREGAR PACOTES ESSENCIAIS
library(embedR)
library(lavaan)
library(psych)
library(readxl)

# 2. CARREGAR OS ITEMS 
url <- "https://raw.githubusercontent.com/FredPedrosa/Generative_Pyschometrics/main/template/itens/itens_escala_final_purificada_com_textos.xlsx"
arquivo_temp <- tempfile(fileext = ".xlsx")
download.file(url, destfile = arquivo_temp, mode = "wb")
dados <- read_excel(arquivo_temp)

# 2.1. Organizar os IDs para bater perfeitamente com a sintaxe do Lavaan
dados$Novo_ID <- sprintf("i%02d", 1:nrow(dados))
dados <- dados[, c("Novo_ID", "ID_Item", "Texto_do_Item", "Dimensao_Teorica", "Cluster_EGA")]
colnames(dados)[2] <- "ID_Original"
dados <- as.data.frame(dados)

# 3. EXTRAIR EMBEDDINGS (COHERE)

# É preciso colocar a chave API cohere aqui
#er_set_token(cohere = "SUA_CHAVE_API_AQUI")

vetor_de_texto <- dados$Texto_do_Item
matriz_embeddings <- er_embed(vetor_de_texto, api = "cohere")
rownames(matriz_embeddings) <- dados$Novo_ID

# 4. GERAR A MATRIZ SINTÉTICA (SPEARMAN)
# Extrair a matriz de correlação direta dos embeddings
matriz_sintetica_pura <- cor(t(matriz_embeddings), method = "spearman")

# 5. ESPECIFICAR O MODELO DE MENSURAÇÃO 
modelo_teste_simples <- '
  Ansiedade =~ i01 + i02 + i03 + i04 + i05 + i06 + i07 + i08 + i09
  Depressao =~ i10 + i11 + i12 + i13 + i14 + i15 + i16 + i17
  Estresse  =~ i18 + i19 + i20 + i21 + i22 + i23 + i24 + i25 + i26 + i27 + i28 + i29
'

# 6. AJUSTE DA PSEUDO-CFA
fit_29_simples <- cfa(
  modelo_teste_simples, 
  sample.cov = matriz_sintetica_pura, 
  sample.nobs = 250, 
  estimator = "ML"
)

# 7. EXTRAÇÃO DAS MÉTRICAS DE AJUSTE
fitMeasures(fit_29_simples, c("chisq", "df", "pvalue", "cfi","tli", "rmsea", 
                              "rmsea.ci.lower", "rmsea.ci.upper", "srmr"))

# 8. RESUMO COMPLETO DO MODELO (Cargas fatoriais)
summary(fit_29_simples, standardized = TRUE)


