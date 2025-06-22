# 🧠 Compilador Pokémon

Um compilador educacional e temático baseado no universo Pokémon, criado com [PLY (Python Lex-Yacc)](https://www.dabeaz.com/ply/) com suporte a:

- ✅ Análise léxica com tokens personalizados (ex: `PLUSLE` para soma, `VAPOREON` para `int`, etc.)
- ✅ Análise sintática com geração de AST
- ✅ Análise semântica com verificação de tipos, escopos e inicialização de variáveis
- ✅ Geração de código intermediário (tradução para 3 endereços)
- ✅ Temática 100% Pokémon: comandos, variáveis, operadores e estruturas

---

## 📜 Exemplo de Código

```pokelang
VAPOREON POKEDEXidade POKEBALL NUMEL20
VAPOREON POKEDEXresultado POKEBALL POKEDEXidade PLUSLE NUMEL5
SMEARGLE POKEDEXresultado
GIRATINA
```

### 🔍 Equivalente em linguagem comum:

```c
int idade = 20;
int resultado = idade + 5;
print(resultado);
```

---

## 🧩 Tokens e Palavras-chave

| Token Pokémon     | Significado            |
|-------------------|------------------------|
| `VAPOREON`        | `int` (tipo inteiro)   |
| `CHARMELEON`      | `char` (tipo caractere)|
| `STRING`          | Strings com `"`        |
| `NUMEL10`         | Números inteiros       |
| `NUMEL3.14`       | Números reais          |
| `PLUSLE`, `MINUN` | Soma e subtração       |
| `MEWTWO`, `SCYTHER` | Multiplicação e divisão |
| `POKEBALL`        | Atribuição (`=`)       |
| `SMEARGLE`        | Print (`print`)        |
| `GREATBALL`, `MASTERBALL` | If / Else       |
| `GIRATINA`        | Fim do programa        |
| `POKEDEXnome`     | Identificador (`nome`) |

---

## 🧠 Análise Semântica

O analisador semântico detecta:

- ❌ Uso de variáveis não declaradas
- ❌ Atribuição de tipos incompatíveis (ex: `char` → `int`)
- ⚠️ Uso de variáveis não inicializadas
- ✅ Verificação de escopos com suporte a blocos `if/else`

---

## ⚙️ Geração de Código Intermediário

Exemplo da saída gerada:

```
1: declare VAPOREON idade
2: t1 = 20
3: idade = t1
4: declare VAPOREON resultado
5: t2 = idade PLUSLE 5
6: resultado = t2
7: print resultado
```

---

## 🚀 Como rodar

### 1. Instale dependências

```bash
pip install ply
```

### 2. Execute o compilador

```bash
python compilador_pokemon.py
```

O script já vem com **testes prontos** para diferentes casos de uso (corretos e com erro).

---

## 📁 Estrutura

```
compilador_pokemon.py    # Código principal
├── análise léxica       # Tokens estilo Pokémon
├── análise sintática    # Parser com regras em PLY
├── análise semântica    # Tabela de símbolos, tipos e escopos
└── geração de código    # Geração de código intermediário
```

---

## 🎓 Objetivos educacionais

Este projeto foi desenvolvido com o objetivo de:

- Aprender os fundamentos da construção de compiladores
- Explorar análise léxica, sintática e semântica com PLY
- Tornar o aprendizado mais divertido com uma temática Pokémon

---

## 👨‍💻 Autor

Desenvolvido por **Heder Santos**, estudante de Ciência da Computação no IF Goiano.

---

## 🧪 Testes inclusos

- ✅ Código válido
- ❌ Variável não declarada
- ❌ Tipos incompatíveis
- ✅ Estruturas `if/else`

---

## 📘 Referências

- [PLY - Python Lex-Yacc](https://www.dabeaz.com/ply/)
- Compiladores: Princípios, Técnicas e Ferramentas (Livro do Dragão)
- [Pokémon API](https://pokeapi.co/) — apenas inspiração 😄

---

## ⭐ Dê uma estrela!

Se curtiu o projeto, deixe uma ⭐ aqui no repositório! Isso ajuda muito 🚀

```
