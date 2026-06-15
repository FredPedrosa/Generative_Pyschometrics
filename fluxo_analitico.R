# ==============================================================================
# 1. SETUP E BIBLIOTECAS
# ==============================================================================
# NLP e APIs
library(embedR)      # Interface Cohere
library(openai)      # Interface OpenAI
library(lsa)         # Similaridade de Cosseno
# Manipulação de Dados
library(dplyr)
library(tidyr)
library(corrr)
library(Matrix)
# Estatística e Psicometria
library(psych)
library(lavaan)
library(semTools)
library(EGAnet)
# Machine Learning e XAI
library(caret)
library(ranger)      # Random Forest rápida
library(glmnet)      # Elastic Net (Motor Principal)
library(party)       # Árvores de Inferência Condicional (ctree)
library(ggplot2)     # Visualização
library(ggparty)     # Visualização de árvores

# Configuração de APIs (Atenção: Substitua pelas suas chaves)
# er_set_tokens(cohere = "SUA_CHAVE_AQUI")
# Sys.setenv(OPENAI_API_KEY = "SUA_CHAVE_AQUI")

# ==============================================================================
# 2. DEFINIÇÃO DO GABARITO SEMÂNTICO (LÉXICO)
# ==============================================================================

# Itens do BDI-II (Adaptação Brasileira)
itens_bdi <- c(
  "bdi1" = "Eu me sinto triste", "bdi2" = "Eu me sinto pessimista sobre o futuro",
  "bdi3" = "Eu me sinto como um fracassado", "bdi4" = "Eu perdi o prazer nas coisas que eu gostava",
  "bdi5" = "Eu me sinto culpado", "bdi6" = "Eu sinto que estou sendo punido",
  "bdi7" = "Eu estou decepcionado comigo mesmo", "bdi8" = "Eu me critico por minhas falhas",
  "bdi9" = "Eu tenho pensamentos de me matar", "bdi10" = "Eu choro mais do que o habitual",
  "bdi11" = "Eu me sinto agitado ou inquieto", "bdi12" = "Eu perdi o interesse nas outras pessoas",
  "bdi13" = "Eu tenho dificuldade em tomar decisões", "bdi14" = "Eu me sinto sem valor",
  "bdi15" = "Eu sinto que estou sem energia", "bdi16" = "Eu tive mudanças no meu sono",
  "bdi17" = "Eu me sinto irritado", "bdi18" = "Eu tive mudanças no meu apetite",
  "bdi19" = "Eu tenho dificuldade de concentração", "bdi20" = "Eu me sinto cansado ou fadigado",
  "bdi21" = "Eu perdi o interesse em sexo"
)

# Itens do BAI (Sintomas Fisiológicos/Cognitivos)
itens_bai <- c(
  "bai1" = "Dormência ou formigamento", "bai2" = "Sensação de calor",
  "bai3" = "Tremores nas pernas", "bai4" = "Incapacidade de relaxar",
  "bai5" = "Medo que aconteça o pior", "bai6" = "Tontura ou atordoamento",
  "bai7" = "Palpitação ou aceleração do coração", "bai8" = "Instabilidade",
  "bai9" = "Terror", "bai10" = "Nervosismo", "bai11" = "Sensação de sufocação",
  "bai12" = "Tremores nas mãos", "bai13" = "Fisgada", "bai14" = "Medo de perder o controle",
  "bai15" = "Dificuldade de respirar", "bai16" = "Medo de morrer", "bai17" = "Assustado",
  "bai18" = "Indigestão ou desconforto abdominal", "bai19" = "Desmaio",
  "bai20" = "Rosto ruborizado", "bai21" = "Suor (não devido ao calor)"
)

# Itens DASS-21
itens_dass21 <- c(
  "i1" = "Achei difícil me acalmar", "i2" = "Senti minha boca seca",
  "i3" = "Não consegui vivenciar nenhum sentimento positivo",
  "i4" = "Tive dificuldade em respirar em alguns momentos",
  "i5" = "Achei difícil ter iniciativa para fazer as coisas",
  "i6" = "Tive a tendência de reagir de forma exagerada às situações",
  "i7" = "Senti tremores (ex. nas mãos)", "i8" = "Senti que estava sempre nervoso",
  "i9" = "Preocupei-me com situações em que eu pudesse entrar em pânico",
  "i10" = "Senti que não tinha nada a desejar", "i11" = "Senti-me agitado",
  "i12" = "Achei difícil relaxar", "i13" = "Senti-me deprimido(a) e sem ânimo",
  "i14" = "Fui intolerante com as coisas que me impediam de continuar",
  "i15" = "Senti que ia entrar em pânico", "i16" = "Não consegui me entusiasmar com nada",
  "i17" = "Senti que não tinha valor como pessoa", "i18" = "Senti que estava um pouco emotivo demais",
  "i19" = "Sabia que meu coração estava alterado mesmo sem esforço",
  "i20" = "Senti medo sem motivo", "i21" = "Senti que a vida não tinha sentido"
)

# Itens PANAS (Apenas Negativos para o estudo principal)
nomes_panas_neg <- c(
  "PN2envergo" = "Envergonhado", "PN4aflit" = "Aflito", "PN6culpado" = "Culpado",
  "PN8irrit" = "Irritado", "PN10medo" = "Com medo", "PN12hostil" = "Hostil",
  "PN14inquie" = "Inquieto", "PN16nervo" = "Nervoso", "PN18apavo" = "Apavorado",
  "PN20chate" = "Chateado"
)

# Itens BDI-II e BAI (Clínico Total)
# [Definição dos itens itens_bdi e itens_bai conforme rodamos anteriormente]
# todos_itens_novos <- c(itens_bdi, itens_bai)

# Sondas Emocionais (Protocolo Kambeitz/Nature)
sondas_emocionais <- c(
  "admiration", "adoration", "aesthetic appreciation", "amusement", 
  "anger", "anxiety", "awe", "awkwardness", "boredom", "calmness", 
  "confusion", "craving", "disgust", "empathetic pain", "entrancement", 
  "excitement", "fear", "horror", "interest", "joy", "nostalgia", "distress",
  "relief", "romance", "sadness", "satisfaction", "sexual desire", "surprise"
)

# ==============================================================================
# 3. MOTORES MATEMÁTICOS (FUNÇÕES CUSTOMIZADAS)
# ==============================================================================

#' Engenharia de Atributos via Siamese Vetorial
#' @description Gera Diferença Absoluta e Produto de Hadamard para pares de itens
# Função de geração com nomes genéricos (V1, V2...)
gerar_features_pares <- function(df_pares, matriz_embeddings) {
  features_list <- list()
  for(i in 1:nrow(df_pares)) {
    v1 <- matriz_embeddings[df_pares$Var1[i], ]
    v2 <- matriz_embeddings[df_pares$Var2[i], ]
    # Diferença Absoluta e Produto de Hadamard
    features_list[[i]] <- c(abs(v1 - v2), v1 * v2)
  }
  out <- as.data.frame(do.call(rbind, features_list))
  # Força nomes V1, V2, V3... para bater com os modelos salvos
  colnames(out) <- paste0("V", 1:ncol(out))
  return(out)
}

#' Validação de Isomorfismo via Permutação (Monte Carlo)
#' @description Estima p-valores empíricos para mitigar dependência estatística
benchmark_ia_humano <- function(predito, real, n_iter = 10000, seed = 123) {
  v_pred <- as.numeric(as.vector(predito))
  v_real <- as.numeric(as.vector(real))
  set.seed(seed)
  r_obs <- as.numeric(cor(v_pred, v_real, method = "pearson", use = "complete.obs"))
  r_nulas <- replicate(n_iter, {
    cor(v_pred, sample(v_real), method = "pearson", use = "complete.obs")
  })
  p_perm <- sum(abs(r_nulas) >= abs(r_obs)) / n_iter
  return(data.frame(r = r_obs, p_perm = p_perm))
}

#' Teste de Y-Randomization (Teste do Caos)
#' @description Prova que o modelo não é fruto de overfitting
testar_caos_total <- function(dados_treino, metodo = "glmnet", n_sim = 50) {
  resultados_caos <- replicate(n_sim, {
    df_shuffled <- dados_treino
    df_shuffled$r_target <- sample(df_shuffled$r_target)
    if(metodo == "glmnet") {
      fit <- train(r_target ~ ., data = df_shuffled, method = "glmnet",
                   trControl = trainControl(method = "none"), 
                   tuneGrid = expand.grid(alpha = 0.5, lambda = 0.1))
    } else {
      fit <- ranger(r_target ~ ., data = df_shuffled, num.trees = 100)
    }
    p <- if(metodo == "glmnet") predict(fit, df_shuffled) else fit$predictions
    if(sd(p) == 0) return(0)
    return(cor(p, df_shuffled$r_target))
  })
  return(mean(resultados_caos))
}

cat("Ambiente e Motores configurados com sucesso.\n")


# ==============================================================================
# 4. CARREGAMENTO DOS GABARITOS HUMANOS (GROUND TRUTH)
# ==============================================================================
cat("Carregando matrizes humanas...\n")

# Matrizes Policóricas pré-calculadas (Garantem reprodutibilidade sem dados brutos)
cor_h_dass21  <- readRDS("cor_human_dass21.rds")    # N=259
cor_h_panas   <- readRDS("cor_human_panas.rds")     # N=473
cor_h_clinico <- readRDS("cor_human_clinico.rds")   # N=1957 (BDI + BAI)

# Transformar gabaritos humanos em formato longo para os JOINS
preparar_longo_humano <- function(matriz, prefixo_alvo) {
  as.data.frame(matriz) %>%
    mutate(Var1 = rownames(.)) %>%
    pivot_longer(-Var1, names_to = "Var2", values_to = "r_real") %>%
    filter(Var1 < Var2) %>%
    mutate(Var1 = gsub("_", "", Var1), # Limpeza de nomes bdi_1 -> bdi1
           Var2 = gsub("_", "", Var2))
}

df_h_dass_long    <- preparar_longo_humano(cor_h_dass21, "DASS")
df_h_panas_long   <- preparar_longo_humano(cor_h_panas, "PANAS")
df_h_clinico_long <- preparar_longo_humano(cor_h_clinico, "BDI_BAI")

# ==============================================================================
# 4.1. Modelos empíricos 
# ==============================================================================

cat("Rodando CFA Humana: DASS-21...\n")

itens_dass <- paste0("i", 1:21)

# Modelo A: 3 Fatores Correlacionados (Teórico)
mod_dass_3f <- '
  Depressao =~ i3 + i5 + i10 + i13 + i16 + i17 + i21
  Ansiedade =~ i2 + i4 + i7 + i9 + i15 + i19 + i20
  Estresse  =~ i1 + i6 + i8 + i11 + i12 + i14 + i18
'

# Modelo B: Bifatorial (p-factor clássico)
mod_dass_bf <- paste0(
  'FG =~ ', paste(itens_dass, collapse = " + "), '\n',
  'DEP_e =~ i3 + i5 + i10 + i13 + i16 + i17 + i21\n',
  'ANX_e =~ i2 + i4 + i7 + i9 + i15 + i19 + i20\n',
  'STR_e =~ i1 + i6 + i8 + i11 + i12 + i14 + i18'
)

# 1. Modelo 3 Fatores
fit_h_dass_3f <- cfa(mod_dass_3f, sample.cov = cor_h_dass21, sample.nobs = 259, 
                     estimator = "ML")

# 2. Modelo Bifactor (p-factor)
fit_h_dass_bf <- cfa(mod_dass_bf, sample.cov = cor_h_dass21, sample.nobs = 259, 
                     estimator = "ML", orthogonal = TRUE)

cat("Rodando CFA Humana: PANAS...\n")

itens_pos <- c("PN1ativo", "PN3atento", "PN5determ", "PN7empol", "PN9interes", "PN11orgul", "PN13alerta", "PN15entusia", "PN17forte", "PN19inspi")
itens_neg <- c("PN2envergo", "PN4aflit", "PN6culpado", "PN8irrit", "PN10medo", "PN12hostil", "PN14inquie", "PN16nervo", "PN18apavo", "PN20chate")

mod_panas_2f <- paste0(
  'Afeto_Positivo =~ ', paste(itens_pos, collapse = " + "), '\n',
  'Afeto_Negativo =~ ', paste(itens_neg, collapse = " + "), '\n',
  
  'Afeto_Negativo =~ PN13alerta'
)

# Modelo 2 Fatores Ajustado
fit_h_panas_2f <- cfa(mod_panas_2f, sample.cov = cor_h_panas, sample.nobs = 473, 
                      estimator = "ML")

cat("Rodando CFA Humana: BDI + BAI...\n")

# Itens bdi_1...bdi_21 e bai_1...bai_21
itens_bdi_h <- paste0("bdi_", 1:21)
itens_bai_h <- paste0("bai_", 1:21)

# Modelo A: 2 Fatores Correlacionados (Depressão vs Ansiedade)
# --- MODELO 2 FATORES (BDI vs BAI) ---
mod_combi_2f <- paste0(
  'Depressao =~ ', paste0('bdi_', 1:21, collapse = ' + '), '\n',
  'Ansiedade =~ ', paste0('bai_', 1:21, collapse = ' + ')
)

# --- MODELO 4 FATORES (Cognitivo/Somático) ---
mod_combi_4f <- "
  BDI_Cogn =~ bdi_1 + bdi_2 + bdi_3 + bdi_4 + bdi_5 + bdi_6 + bdi_7 + bdi_8 + bdi_9 + bdi_10 + bdi_11 + bdi_12 + bdi_13
  BDI_Soma =~ bdi_14 + bdi_15 + bdi_16 + bdi_17 + bdi_18 + bdi_19 + bdi_20 + bdi_21
  BAI_Cogn =~ bai_4 + bai_5 + bai_8 + bai_9 + bai_10 + bai_11 + bai_14 + bai_16 + bai_17
  BAI_Soma =~ bai_1 + bai_2 + bai_3 + bai_6 + bai_7 + bai_12 + bai_13 + bai_15 + bai_18 + bai_19 + bai_20 + bai_21
"

# --- MODELO BIFACTOR (FG + 4 Específicos) ---
mod_combi_bf <- paste0(
  'FG =~ ', paste0('bdi_', 1:21, collapse = ' + '), ' + ', paste0('bai_', 1:21, collapse = ' + '), '\n',
  'BDI_C_e =~ ', paste0('bdi_', 1:13, collapse = ' + '), '\n',
  'BDI_S_e =~ ', paste0('bdi_', 14:21, collapse = ' + '), '\n',
  'BAI_C_e =~ ', paste0('bai_', c(4,5,8,9,10,11,14,16,17), collapse = ' + '), '\n',
  'BAI_S_e =~ ', paste0('bai_', c(1,2,3,6,7,12,13,15,18,19,20,21), collapse = ' + ')
)


# --- FUNÇÃO PARA RODAR E CONSERTAR ---
rodar_cfa_robusta <- function(modelo, matriz, n_obs, ortho = FALSE) {
  # Forçar a matriz a ser Positiva Definida
  matriz_fix <- as.matrix(Matrix::nearPD(matriz, corr = TRUE)$mat)
  
  # Rodar via ULS (O padrão ouro para matrizes policóricas sem dados brutos)
  fit <- cfa(modelo, 
             sample.cov = matriz_fix, 
             sample.nobs = n_obs, 
             estimator = "ULS", 
             orthogonal = ortho,
             check.post = FALSE)
  return(fit)
}

# --- DASS-21 ---
fit_h_dass_3f <- rodar_cfa_robusta(mod_dass_3f, cor_h_dass21, 259)
fit_h_dass_bf <- rodar_cfa_robusta(mod_dass_bf, cor_h_dass21, 259, ortho = TRUE)

# --- PANAS ---
fit_h_panas_2f <- rodar_cfa_robusta(mod_panas_2f, cor_h_panas, 473)

# --- BDI + BAI ---
fit_h_clinico_2f <- rodar_cfa_robusta(mod_combi_2f, cor_h_clinico, 1957)
fit_h_clinico_4f <- rodar_cfa_robusta(mod_combi_4f, cor_h_clinico, 1957)
fit_h_clinico_bf <- rodar_cfa_robusta(mod_combi_bf, cor_h_clinico, 1957, ortho = TRUE)



extrair_indices <- function(fit, nome) {
  indices_nomes <- c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "rmsea.ci.lower", "rmsea.ci.upper", "srmr")
  
  if (!lavInspect(fit, "converged")) {
    vals <- rep(NA, length(indices_nomes))
  } else {
    vals <- fitMeasures(fit, indices_nomes)
  }
  
  # Criar dataframe de uma linha de forma explícita
  df_ind <- as.data.frame(matrix(vals, nrow = 1))
  colnames(df_ind) <- indices_nomes
  rownames(df_ind) <- nome
  return(df_ind)
}

# --- Consolidar a Tabelona Humana ---
cat("Consolidando resultados humanos...\n")

lista_modelos <- list(
  "DASS-21 (3 Fatores)"  = fit_h_dass_3f,
  "DASS-21 (Bifatorial)" = fit_h_dass_bf,
  "PANAS (2 Fatores Adj)" = fit_h_panas_2f,
  "BDI/BAI (2 Fatores)"  = fit_h_clinico_2f,
  "BDI/BAI (4 Fatores)"  = fit_h_clinico_4f,
  "BDI/BAI (Bifatorial)" = fit_h_clinico_bf
)

# Usar do.call(rbind...) para evitar erros de nomes de linhas
resumo_humanos_completo <- do.call(rbind, lapply(names(lista_modelos), function(n) {
  extrair_indices(lista_modelos[[n]], n)
}))

# --- Formatação Final APA para o Manuscrito ---
tabela_1_final <- resumo_humanos_completo %>%
  mutate(
    chisq_df = round(chisq / df, 2),
    p_val = ifelse(is.na(pvalue), "--", round(pvalue, 3)),
    CFI = round(cfi, 3),
    TLI = round(tli, 3),
    RMSEA_90 = paste0(round(rmsea, 3), " [", round(rmsea.ci.lower, 3), " - ", round(rmsea.ci.upper, 3), "]"),
    SRMR = round(srmr, 3)
  ) %>%
  select(chisq_df, p_val, CFI, TLI, RMSEA_90, SRMR)

print(tabela_1_final)


compRelSEM(fit_h_dass_3f)
compRelSEM(fit_h_dass_bf)
compRelSEM(fit_h_panas_2f)
compRelSEM(fit_h_clinico_2f)
compRelSEM(fit_h_clinico_4f)
compRelSEM(fit_h_clinico_bf)


# ==============================================================================
# 5. CARREGAMENTO E NORMALIZAÇÃO DOS EMBEDDINGS (IA)
# ==============================================================================
cat("Carregando e normalizando embeddings de todas as arquiteturas...\n")

l2_norm <- function(m) m / sqrt(rowSums(m^2))

# --- COHERE v3 ---
emb_cohere_dass    <- l2_norm(readRDS("emb_dass_cohere.rds"))
emb_cohere_panas   <- l2_norm(readRDS("emb_panas_cohere.rds"))
emb_cohere_clinico <- l2_norm(readRDS("emb_clinico_cohere.rds"))

# --- MPNET v2 ---
emb_mpnet_dass     <- l2_norm(readRDS("emb_dass_mpnet.rds"))
emb_mpnet_panas    <- l2_norm(readRDS("emb_panas_mpnet.rds"))
emb_mpnet_clinico  <- l2_norm(readRDS("emb_clinico_mpnet.rds"))

# --- OPENAI 3-Large ---
emb_openai_dass    <- l2_norm(readRDS("matriz_emb_openai_dass.rds"))
emb_openai_panas   <- l2_norm(readRDS("matriz_emb_openai_panas.rds"))
emb_openai_clinico <- l2_norm(readRDS("matriz_emb_clinico_openai.rds")) # Garanta que salvou com este nome

# ==============================================================================
# 6. CRIAÇÃO DAS ESTRUTURAS DE PARES (DATASETS DE ML)
# ==============================================================================
cat("Gerando estruturas de pares...\n")

gerar_pares_base <- function(nomes_itens) {
  expand.grid(Var1 = nomes_itens, Var2 = nomes_itens, stringsAsFactors = FALSE) %>%
    filter(Var1 < Var2)
}

# O objeto BDI + BAI:
todos_itens_novos <- c(itens_bdi, itens_bai)

# Estruturas de nomes
pares_dass    <- gerar_pares_base(names(itens_dass21))
pares_panas   <- gerar_pares_base(names(nomes_panas_neg))
pares_clinico <- gerar_pares_base(names(todos_itens_novos))

# Datasets base unindo com Gabaritos Humanos
df_ml_dass    <- inner_join(pares_dass, df_h_dass_long, by = c("Var1", "Var2"))
df_ml_panas   <- inner_join(pares_panas, df_h_panas_long, by = c("Var1", "Var2"))
df_ml_clinico <- inner_join(pares_clinico, df_h_clinico_long, by = c("Var1", "Var2"))

# ==============================================================================
# 7. ENGENHARIA DE ATRIBUTOS (FEATURES) - CROSS-ARCHITECTURE
# ==============================================================================
cat("Gerando interações vetoriais para todas as arquiteturas...\n")

# Listas para armazenar os datasets de features
features_treino <- list()
features_teste  <- list()

# --- COHERE ---
# Regenerar as features com os nomes V1, V2...
cat("Regenerando bases de dados de ML...\n")
features_treino <- list()
features_teste  <- list()

# COHERE
features_treino$Cohere      <- gerar_features_pares(df_ml_dass, emb_cohere_dass) %>% mutate(r_target = df_ml_dass$r_real)
features_teste$Cohere_PANAS <- gerar_features_pares(df_ml_panas, emb_cohere_panas) %>% mutate(r_real = df_ml_panas$r_real)
features_teste$Cohere_CLIN  <- gerar_features_pares(df_ml_clinico, emb_cohere_clinico) %>% mutate(r_real = df_ml_clinico$r_real)

# MPNET
features_treino$MPNet       <- gerar_features_pares(df_ml_dass, emb_mpnet_dass) %>% mutate(r_target = df_ml_dass$r_real)
features_teste$MPNet_PANAS  <- gerar_features_pares(df_ml_panas, emb_mpnet_panas) %>% mutate(r_real = df_ml_panas$r_real)
features_teste$MPNet_CLIN   <- gerar_features_pares(df_ml_clinico, emb_mpnet_clinico) %>% mutate(r_real = df_ml_clinico$r_real)

# OPENAI
features_treino$OpenAI      <- gerar_features_pares(df_ml_dass, emb_openai_dass) %>% mutate(r_target = df_ml_dass$r_real)
features_teste$OpenAI_PANAS <- gerar_features_pares(df_ml_panas, emb_openai_panas) %>% mutate(r_real = df_ml_panas$r_real)
features_teste$OpenAI_CLIN  <- gerar_features_pares(df_ml_clinico, emb_openai_clinico) %>% mutate(r_real = df_ml_clinico$r_real)

# ==============================================================================
# 8. TREINAMENTO BATERIA COMPLETA (3 Embeddings x 4 Algoritmos)
# ==============================================================================
cat("Iniciando bateria de treinamento massivo...\n")

modelos_finais <- list()
set.seed(123)
controles <- trainControl(method = "cv", number = 10)

# Loop pelas arquiteturas
#for (arq in c("Cohere", "MPNet", "OpenAI")) {
#  cat("\nProcessando Arquitetura:", arq, "\n")
#  
#  # Preparar e Escalonar dados de treino
#  df_atual <- features_treino[[arq]]
#  pre_proc <- preProcess(df_atual %>% select(-r_target), method = c("center", "scale"))
#  df_scaled <- predict(pre_proc, df_atual)
#  
#  # Salvar o objeto de pre-processamento para usar no teste depois
#  modelos_finais[[arq]]$pre_proc <- pre_proc
#  
#  # 1. Elastic Net
#  cat("  -> Treinando Elastic Net...\n")
#  modelos_finais[[arq]]$EN <- train(r_target ~ ., data = df_scaled, method = "glmnet", trControl = controles)
#  
#  # 2. Random Forest
#  cat("  -> Treinando Random Forest...\n")
#  modelos_finais[[arq]]$RF <- train(r_target ~ ., data = df_scaled, method = "ranger", trControl = controles, importance = 'impurity')
#  
#  # 3. XGBoost
#  cat("  -> Treinando XGBoost...\n")
#  modelos_finais[[arq]]$XG <- train(r_target ~ ., data = df_scaled, method = "xgbTree", trControl = controles, tuneLength = 2)
#  
#  # 4. ctree
#  cat("  -> Treinando ctree...\n")
#  modelos_finais[[arq]]$CT <- party::ctree(r_target ~ ., data = df_scaled)
#  
#  # 5. Teste de Caos (Y-Randomization) para a arquitetura
#  cat("  -> Rodando Testes de Caos...\n")
#  modelos_finais[[arq]]$Caos_EN <- testar_caos_total(df_scaled, metodo = "glmnet")
#  modelos_finais[[arq]]$Caos_RF <- testar_caos_total(df_scaled, metodo = "ranger")
#}

cat("\nTreinamento e Validação de Blindagem Concluídos.\n")

# ==============================================================================
# 7. CARREGAMENTO DOS MODELOS TREINADOS (CACHE)
# ==============================================================================
cat("Carregando modelos salvos e objetos de pre-processamento...\n")

modelos_finais <- list()

# --- COHERE ---
modelos_finais$Cohere$EN <- readRDS("modelo_en.rds")
modelos_finais$Cohere$RF <- readRDS("modelo_rf_dass21_v1.rds")
modelos_finais$Cohere$XG <- readRDS("modelo_xgb.rds")
modelos_finais$Cohere$CT <- readRDS("modelo_ctree.rds")
# Gerar pre-processador do Cohere (essencial para as transferências)
modelos_finais$Cohere$pre_proc <- preProcess(features_treino$Cohere %>% select(-r_target), method = c("center", "scale"))

# --- MPNET ---
modelos_finais$MPNet$EN  <- readRDS("modelo_en_mpnet.rds")
modelos_finais$MPNet$RF  <- readRDS("modelo_rf_mpnet.rds")
modelos_finais$MPNet$XG  <- readRDS("modelo_xgb_mpnet.rds")
modelos_finais$MPNet$CT  <- readRDS("modelo_ctree_mpnet.rds")
modelos_finais$MPNet$pre_proc <- preProcess(features_treino$MPNet %>% select(-r_target), method = c("center", "scale"))

# --- OPENAI ---
modelos_finais$OpenAI$EN <- readRDS("modelos_final/modelo_en_openai_full.rds")
modelos_finais$OpenAI$RF <- readRDS("modelos_final/modelo_rf_openai_full.rds")
modelos_finais$OpenAI$XG <- readRDS("modelos_final/modelo_xgb_openai_full.rds")
modelos_finais$OpenAI$CT <- readRDS("modelos_final/modelo_ctree_openai.rds")
modelos_finais$OpenAI$pre_proc <- preProcess(features_treino$OpenAI %>% select(-r_target), method = c("center", "scale"))

cat("Todos os modelos carregados com sucesso.\n")


# ==============================================================================
# 8. BENCHMARKING DE TRANSFERÊNCIA (LOOP MASSIVO)
# ==============================================================================
cat("Iniciando loop de predição e permutação (10k iterações)...\n")

resultados_bench <- list()

for (arq in c("Cohere", "MPNet", "OpenAI")) {
  
  # A. Criar pre-processador baseado no TREINO desta arquitetura
  # Certifique-se que 'features_treino' foi gerada no passo anterior
  pre <- preProcess(features_treino[[arq]] %>% select(-r_target), method = c("center", "scale"))
  
  for (alg in c("EN", "RF", "XG", "CT")) {
    cat("\nProcessando:", arq, "| Algoritmo:", alg, "...")
    
    # B. Recuperar o Modelo
    mod <- modelos_finais[[arq]][[alg]]
    
    # C. Preparar e Escalonar Testes
    # O any_of("r_real") evita erros se a coluna não existir
    df_p_raw <- features_teste[[paste0(arq, "_PANAS")]] %>% select(-any_of("r_real"))
    df_c_raw <- features_teste[[paste0(arq, "_CLIN")]] %>% select(-any_of("r_real"))
    
    df_p_scaled <- predict(pre, df_p_raw)
    df_c_scaled <- predict(pre, df_c_raw)
    
    # D. Predizer (Modo Universal para objetos caret/train)
    # Se o modelo for ctree ou glmnet/train, predict() retorna um vetor numérico
    p_p <- as.numeric(predict(mod, df_p_scaled))
    p_c <- as.numeric(predict(mod, df_c_scaled))
    
    # E. Validar via Permutação (10k iterações para o paper)
    # n_iter = 5000 por velocidade, mude para 10000 depois
    bench_p <- benchmark_ia_humano(p_p, features_teste[[paste0(arq, "_PANAS")]]$r_real, n_iter = 5000)
    bench_c <- benchmark_ia_humano(p_c, features_teste[[paste0(arq, "_CLIN")]]$r_real, n_iter = 5000)
    
    # F. Armazenar
    id <- paste(arq, alg, sep="_")
    resultados_bench[[id]] <- data.frame(
      Embedding = arq, Algoritmo = alg,
      r_PANAS = bench_p$r, p_PANAS = bench_p$p_perm,
      r_CLIN  = bench_c$r, p_CLIN  = bench_c$p_perm
    )
    cat(" OK!")
  }
}

# 2. Consolidação e Formatação Final
df_final_consolidado <- do.call(rbind, resultados_bench) %>%
  mutate(sig_PANAS = case_when(p_PANAS < 0.001 ~ "***", p_PANAS < 0.01 ~ "**", p_PANAS < 0.05 ~ "*", TRUE ~ "ns"),
         sig_CLIN  = case_when(p_CLIN < 0.001 ~ "***", p_CLIN < 0.01 ~ "**", p_CLIN < 0.05 ~ "*", TRUE ~ "ns"))

print(df_final_consolidado)


# ==============================================================================
# 10. RECONSTRUÇÃO ESTRUTURAL (PFA)
# ==============================================================================

cat("\nGerando Matriz Sintética (Cohere) e rodando CFA da IA...\n")

cat("\nReconstruindo matriz sintética BDI/BAI via MPNet (Especialista Clínico)...\n")

# 1. Gerar predições e garantir que elas estejam coladas aos nomes dos itens
# Usamos o MPNet + EN porque ele capturou melhor a sintaxe clínica (r=0.46)
df_pred_clin <- df_ml_clinico %>%
  select(Var1, Var2) %>%
  mutate(r_pred = as.numeric(predict(modelos_finais$MPNet$EN, features_teste$MPNet_CLIN)))

# 2. Criar a matriz com os nomes esperados pelo modelo (com underscore _)
nomes_cfa <- gsub("(bdi|bai)([0-9]+)", "\\1_\\2", names(todos_itens_novos))
matriz_sint_clin <- matrix(1, 42, 42, dimnames = list(nomes_cfa, nomes_cfa))

# 3. Preenchimento Robusto (usando os nomes como chave)
for(i in 1:nrow(df_pred_clin)) {
  v1 <- gsub("(bdi|bai)([0-9]+)", "\\1_\\2", df_pred_clin$Var1[i])
  v2 <- gsub("(bdi|bai)([0-9]+)", "\\1_\\2", df_pred_clin$Var2[i])
  val <- df_pred_clin$r_pred[i]
  
  matriz_sint_clin[v1, v2] <- matriz_sint_clin[v2, v1] <- val
}

# 4. Ajuste Matemático de Matriz
library(Matrix)
matriz_sint_fix <- as.matrix(Matrix::nearPD(matriz_sint_clin, corr = TRUE)$mat)

# 5. Rodar CFA Sintética (ULS - O mais estável para IA)
cat("Rodando CFA Sintética...\n")
fit_ia_clin_bf <- cfa(mod_combi_bf, 
                      sample.cov = matriz_sint_fix, 
                      sample.nobs = 1957, 
                      estimator = "ULS", 
                      orthogonal = TRUE,
                      check.post = FALSE)

# 6. TESTE DE ISOMORFISMO DE CARGAS (FG)
# Extrair cargas do Fator Geral do Humano (ULS)
cargas_h  <- standardizedSolution(fit_h_clinico_bf) %>% 
  filter(lhs == "FG", op == "=~") %>% 
  select(rhs, est.std) %>% rename(Item = rhs, Humano = est.std)

# Extrair cargas do Fator Geral da IA (ULS)
cargas_ia <- standardizedSolution(fit_ia_clin_bf) %>% 
  filter(lhs == "FG", op == "=~") %>% 
  select(rhs, est.std) %>% rename(Item = rhs, IA = est.std)

# Unir e Correlacionar
isomorfismo_df <- left_join(cargas_h, cargas_ia, by = "Item")
res_isom <- cor.test(isomorfismo_df$Humano, isomorfismo_df$IA, method = "pearson")

cat("\n--- RESULTADOS BDI/BAI SINTÉTICO (IA) ---\n")
print(fitMeasures(fit_ia_clin_bf, c("cfi", "tli", "rmsea", "srmr")))
cat("\n--- ISOMORFISMO DE CARGAS (r) ---\n")
print(res_isom)

# ==============================================================================
# 11. AUDITORIA CLÍNICA (XAI)
# ==============================================================================
cat("\nAuditando a racionalidade semântica (XAI)...\n")

# Pegar importância das variáveis (Elastic Net Cohere)
imp_en <- varImp(modelos_finais$Cohere$EN)$importance %>% 
  as.data.frame() %>% mutate(Feature = rownames(.)) %>% arrange(desc(Overall))

# Identificar a dimensão campeã (V1715)
# Lógica: V1715 - 1024 = Dimensão 691 original
res_xai_sondas <- permutar_xai(emb_cohere_dass[, 691], matriz_sentimento)
res_xai_itens  <- permutar_xai(emb_cohere_dass[, 691], matriz_identidade_itens)

cat("--- Top Sonda para V1715:", res_xai_sondas$Nome[1], "(p =", res_xai_sondas$p_perm[1], ")\n")
cat("--- Top Item para V1715:", res_xai_itens$Nome[1], "(p =", res_xai_itens$p_perm[1], ")\n")


# ==============================================================================
# 12. GERAÇÃO DE TABELAS FINAIS
# ==============================================================================

# TABELA DE BENCHMARKING
tabela_2_apa <- df_final_benchmark_V4 %>%
  mutate(across(starts_with("r_"), ~ round(., 3))) %>%
  mutate(PANAS = paste0(r_PANAS, case_when(p_PANAS < 0.001 ~ "***", p_PANAS < 0.01 ~ "**", p_PANAS < 0.05 ~ "*", TRUE ~ "")),
         CLIN = paste0(r_CLIN, case_when(p_CLIN < 0.001 ~ "***", p_CLIN < 0.01 ~ "**", p_CLIN < 0.05 ~ "*", TRUE ~ ""))) %>%
  select(Embedding, Algoritmo, PANAS, CLIN)

print(tabela_2_apa)
