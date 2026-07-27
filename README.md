# A Topologia do Sofrimento Psíquico: Calibração de Instrumentos Psicométricos via Inteligência Artificial Explicável 🧠🤖

Este repositório hospeda os materiais suplementares, conjuntos de dados, modelos preditivos e o protocolo metodológico oficial desenvolvidos no estudo sobre Psicometria Generativa.

## 🎯 Objetivo do Estudo

O estudo demonstra que a estrutura latente dos transtornos mentais é uma propriedade emergente codificada na semântica da linguagem. Ao integrar *Large Language Models* (LLMs) à modelagem psicométrica, estabelecemos um benchmark em relação às possibilidades da **Psicometria Generativa** no campo da psicopatologia — um paradigma que permite mapear eixos semiológicos fundamentais e realizar a calibração pré-empírica de instrumentos psicológicos de forma 100% *in silico* por meio do  protocolo proposto de (Generative Confirmatory Factor Analysis - Gen-CFA).

## 📂 Estrutura do Repositório

O repositório está organizado para facilitar a reprodutibilidade das análises e a aplicação do protocolo em novos estudos:

*   📄 **`PsicoGenerativa_Analise.pdf` & `.Rmd`**: Manuscrito do estudo e o script contendo todas as análises originais, incluindo Extração de Atributos, *Explainable AI* (XAI) e *Pseudo-Factor Analysis* (PFA).
*   📄 **`TabelaS1.docx`**: Material suplementar referenciado no artigo.
*   📁 **`embeddings/`**: Matrizes de *embeddings* (vetores densos) extraídas via LLMs utilizadas no treinamento dos modelos.
*   📁 **`modelos_preditivos/`**: Modelos de *Machine Learning* (como o Elastic Net) treinados para prever a covariação de respostas humanas.
*   📁 **`template/`**: Contém o fluxo de trabalho pronto para uso do Protocolo Gen-CFA.
    *   📄 `template.R`: Script automatizado para purificação e validação estrutural de novos itens.
    *   📂 **`itens/`**:
        *   `geração.txt`: *Prompt* e parâmetros utilizados para a criação sintética dos itens.
        *   `itens_novos.xlsx`: Banco bruto de itens gerados pela IA antes da purificação.
        *   `itens_escala_final_purificada_com_textos.xlsx`: Escala final após a triagem estrutural via EGA, com os 29 itens retidos e validados.

## 🛠️ O Protocolo Gen-CFA (Generative CFA)

O Gen-CFA é um fluxo de trabalho em ambiente R (disponível na pasta `/template`) criado para auxiliar pesquisadores na validação pré-empírica de escalas. 

Ele permite que você teste o ajuste de um novo corpus de itens antes de aplicá-los em humanos com **Pseudo-CFA baseada em correlação de Spearman** para verificar o ajuste da teoria à representação vetorial.

Para utilizar, basta abrir o `template.R` e seguir as instruções. O script já inclui um verificador automático para instalar os pacotes essenciais (`lavaan`, `embedR`, `semTools`, etc.).

## 📜 Como Citar

Se você utilizar o protocolo AI-GENIE ou qualquer insight/código deste material, por favor, cite como:

**Pedrosa, F. (2026). A Topologia do Sofrimento Psíquico: Calibração de Instrumentos Psicométricos via Inteligência Artificial Explicável. Repositório oficial. Disponível em: https://github.com/FredPedrosa/Generative_Pyschometrics**

## ⚖️ Licença

Este projeto está sob a licença **MIT** - o que significa que o código e o protocolo são abertos e livres para uso acadêmico e comercial, desde que a autoria original seja devidamente creditada.
