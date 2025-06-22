import ply.lex as lex
import ply.yacc as yacc

# Lista de tokens definida globalmente
tokens = (
    'SOMA', 'SUB', 'MULT', 'DIV', 'NUM', 'FLOAT', 'ATRIB', 
    'ID', 'FIM', 'COMMENT', 'IF', 'ELSE', 'INT', 'CHAR', 'PRINT',
    'STRING', 'CHAR_LITERAL', 'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'EQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE', 'INPUT'
)


# Definições dos tokens e expressões regulares...
t_SOMA   = r'PLUSLE'
t_SUB    = r'MINUN'
t_MULT   = r'MEWTWO'
t_DIV    = r'SCYTHER'
t_ATRIB  = r'POKEBALL'
t_FIM    = r'GIRATINA'
t_IF     = r'GREATBALL'
t_ELSE   = r'MASTERBALL'
t_INT    = r'VAPOREON'
t_CHAR = r'CHARMELEON'
t_PRINT  = r'SMEARGLE'
t_INPUT  = r'PIKACHU'
t_ignore = ' \t\n'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_LBRACE  = r'\{'
t_RBRACE  = r'\}'
t_EQ  = r'EEVEE'
t_NEQ = r'UMBREON'
t_LT  = r'FLAREON'
t_GT  = r'JOLTEON'
t_LTE = r'ESPEON'
t_GTE = r'SYLVIEON'

def t_STRING(t):
    r'\"([^"\n]*)\"'
    t.value = t.value[1:-1]
    return t

def t_CHAR_LITERAL(t):
    r"CHARMELEON'(.)'"
    t.value = t.value[11:-1]  # Pula "CHARMELEON'" e pega o char
    return t


def t_FLOAT(t):
    r'NUMEL\d+\.\d+'
    t.value = float(t.value[5:])
    return t

def t_COMMENT(t):
    r'ALTARIA.*'
    pass

def t_NUM(t):
    r'NUMEL\d+'
    t.value = int(t.value[5:])
    return t

def t_ID(t):
    r'POKEDEX[a-zA-Z_][a-zA-Z0-9_]*'
    t.value = t.value[7:]
    return t

def t_error(t):
    print(f"Emoji ou caractere inválido '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# ==================== ANÁLISE SEMÂNTICA ====================

class SemanticError(Exception):
    """Exceção customizada para erros semânticos"""
    def __init__(self, message, line=None):
        self.message = message
        self.line = line
        super().__init__(self.message)

class Symbol:
    """Representa um símbolo na tabela de símbolos"""
    def __init__(self, name, symbol_type, data_type, value=None, initialized=False, line=None):
        self.name = name
        self.symbol_type = symbol_type  # 'variable', 'function', etc.
        self.data_type = data_type      # 'VAPOREON', 'CHARMELEON', etc.
        self.value = value
        self.initialized = initialized
        self.line = line
    
    def __repr__(self):
        return f"Symbol({self.name}: {self.data_type}, init={self.initialized})"

class SymbolTable:
    """Tabela de símbolos com suporte a escopos aninhados"""
    def __init__(self):
        self.scopes = [{}]  # Pilha de escopos
        self.current_scope = 0
    
    def enter_scope(self):
        """Entra em um novo escopo"""
        self.scopes.append({})
        self.current_scope += 1
    
    def exit_scope(self):
        """Sai do escopo atual"""
        if self.current_scope > 0:
            self.scopes.pop()
            self.current_scope -= 1
    
    def declare(self, name, symbol_type, data_type, value=None, line=None):
        """Declara um novo símbolo no escopo atual"""
        current_scope = self.scopes[self.current_scope]
        
        if name in current_scope:
            raise SemanticError(f"Variável '{name}' já foi declarada neste escopo", line)
        
        symbol = Symbol(name, symbol_type, data_type, value, False, line)
        current_scope[name] = symbol
        return symbol
    
    def lookup(self, name):
        """Procura um símbolo em todos os escopos (do mais interno ao mais externo)"""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def set_initialized(self, name, value=None):
        """Marca um símbolo como inicializado"""
        symbol = self.lookup(name)
        if symbol:
            symbol.initialized = True
            if value is not None:
                symbol.value = value
            return True
        return False
    
    def get_all_symbols(self):
        """Retorna todos os símbolos de todos os escopos"""
        all_symbols = {}
        for i, scope in enumerate(self.scopes):
            all_symbols[f"scope_{i}"] = scope
        return all_symbols

class TypeChecker:
    """Verificador de tipos para o compilador Pokemon"""
    
    # Mapeamento dos tipos Pokemon para tipos internos
    TYPE_MAPPING = {
        'VAPOREON': 'int',
        'CHARMELEON': 'char',
        'STRING': 'string',
    }

    # Operadores aritméticos válidos por tipo
    ARITHMETIC_OPS = ['PLUSLE', 'MINUN', 'MEWTWO', 'SCYTHER']
    COMPARISON_OPS = ['EEVEE', 'UMBREON', 'FLAREON', 'JOLTEON', 'ESPEON', 'SYLVIEON']
    
    @staticmethod
    def normalize_type(data_type):
        """Converte tipo Pokemon para tipo interno padronizado"""
        if data_type in TypeChecker.TYPE_MAPPING:
            return TypeChecker.TYPE_MAPPING[data_type]
        elif data_type in ['int', 'float', 'string', 'char', 'boolean']:
            return data_type
        else:
            return data_type
    
    @staticmethod
    def can_convert(from_type, to_type):
        """Verifica se um tipo pode ser convertido para outro"""
        # Normaliza os tipos para comparação
        from_normalized = TypeChecker.normalize_type(from_type)
        to_normalized = TypeChecker.normalize_type(to_type)
        
        if from_normalized == to_normalized:
            return True
        
        # Conversões implícitas permitidas
        implicit_conversions = [
            ('int', 'float'),
            # ('char', 'int'),
            ('int', 'string')  # Para prints
        ]
        
        return (from_normalized, to_normalized) in implicit_conversions
    
    @staticmethod
    def get_result_type(left_type, right_type, operator):
        """Determina o tipo resultante de uma operação"""
        left_norm = TypeChecker.normalize_type(left_type)
        right_norm = TypeChecker.normalize_type(right_type)
        
        if operator in TypeChecker.ARITHMETIC_OPS:
            # Para operações aritméticas
            if left_norm == right_norm:
                return left_type  # Retorna o tipo original
            elif TypeChecker.can_convert(left_norm, right_norm):
                return right_type
            elif TypeChecker.can_convert(right_norm, left_norm):
                return left_type
            else:
                return None
        
        elif operator in TypeChecker.COMPARISON_OPS:
            # Comparações sempre retornam boolean
            if left_norm == right_norm or \
               TypeChecker.can_convert(left_norm, right_norm) or \
               TypeChecker.can_convert(right_norm, left_norm):
                return 'boolean'
            else:
                return None
        
        return None
    
    @staticmethod
    def is_valid_operation(data_type, operator):
        """Verifica se uma operação é válida para um tipo específico"""
        normalized = TypeChecker.normalize_type(data_type)
        
        if normalized in ['string', 'char'] and operator in ['MEWTWO', 'SCYTHER']:
            return False  # Não pode multiplicar/dividir strings ou chars
        return True

class SemanticAnalyzer:
    """Analisador semântico principal"""
    
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.type_checker = TypeChecker()
        self.errors = []
        self.warnings = []
    
    def add_error(self, message, line=None):
        """Adiciona um erro à lista de erros"""
        error_msg = f"Linha {line}: {message}" if line else message
        self.errors.append(error_msg)
    
    def add_warning(self, message, line=None):
        """Adiciona um aviso à lista de avisos"""
        warning_msg = f"Linha {line}: {message}" if line else message
        self.warnings.append(warning_msg)
    
    def get_expression_type(self, expr_node):
        """Determina o tipo de uma expressão"""
        if not expr_node:
            return None
        
        node_type = expr_node[0]
        
        if node_type == "number":
            value = expr_node[1]
            return 'int' if isinstance(value, int) else 'float'
        
        elif node_type == "string":
            return 'string'
        
        elif node_type == "char":
            return 'char'
        
        elif node_type == "id":
            var_name = expr_node[1]
            symbol = self.symbol_table.lookup(var_name)
            
            if not symbol:
                self.add_error(f"Variável '{var_name}' não foi declarada")
                return None
            
            if not symbol.initialized:
                self.add_warning(f"Variável '{var_name}' pode não ter sido inicializada")
            
            return symbol.data_type
        
        elif node_type == "binop":
            operator = expr_node[1]
            left_expr = expr_node[2]
            right_expr = expr_node[3]
            
            left_type = self.get_expression_type(left_expr)
            right_type = self.get_expression_type(right_expr)
            
            if not left_type or not right_type:
                return None
            
            # Verifica se a operação é válida para os tipos
            if not self.type_checker.is_valid_operation(left_type, operator):
                self.add_error(f"Operação '{operator}' não é válida para tipo '{left_type}'")
                return None
            
            if not self.type_checker.is_valid_operation(right_type, operator):
                self.add_error(f"Operação '{operator}' não é válida para tipo '{right_type}'")
                return None
            
            # Determina o tipo resultante
            result_type = self.type_checker.get_result_type(left_type, right_type, operator)
            
            if not result_type:
                self.add_error(f"Tipos incompatíveis: '{left_type}' {operator} '{right_type}'")
                return None
            
            return result_type
        
        elif node_type == "condition":
            operator = expr_node[1]
            left_expr = expr_node[2]
            right_expr = expr_node[3]
            
            left_type = self.get_expression_type(left_expr)
            right_type = self.get_expression_type(right_expr)
            
            if not left_type or not right_type:
                return None
            
            result_type = self.type_checker.get_result_type(left_type, right_type, operator)
            
            if not result_type:
                self.add_error(f"Tipos incompatíveis na comparação: '{left_type}' {operator} '{right_type}'")
                return None
            
            return 'boolean'
        
        return None
    
    def analyze_statement(self, stmt):
        """Analisa um statement individual"""
        if not stmt:
            return
        
        stmt_type = stmt[0]
        
        if stmt_type == "vardecl":
            # Declaração de variável: type ID ATRIB expression
            var_type = stmt[1]
            var_name = stmt[2]
            init_expr = stmt[3]
            
            try:
                # Declara a variável
                symbol = self.symbol_table.declare(var_name, 'variable', var_type)
                
                # Verifica o tipo da expressão de inicialização
                expr_type = self.get_expression_type(init_expr)
                
                if expr_type:
                    # Verifica compatibilidade de tipos
                    if not self.type_checker.can_convert(expr_type, var_type):
                        self.add_error(f"Não é possível atribuir '{expr_type}' a variável do tipo '{var_type}'")
                    else:
                        # Marca como inicializada SE a expressão for válida
                        self.symbol_table.set_initialized(var_name)
                
            except SemanticError as e:
                self.add_error(str(e))
        
        elif stmt_type == "assign":
            # Atribuição: ID ATRIB expression
            var_name = stmt[1]
            expr = stmt[2]
            
            symbol = self.symbol_table.lookup(var_name)
            
            if not symbol:
                self.add_error(f"Variável '{var_name}' não foi declarada")
                return
            
            expr_type = self.get_expression_type(expr)
            
            if expr_type:
                if not self.type_checker.can_convert(expr_type, symbol.data_type):
                    self.add_error(f"Não é possível atribuir '{expr_type}' a variável do tipo '{symbol.data_type}'")
                else:
                    self.symbol_table.set_initialized(var_name)
        
        elif stmt_type == "print":
            # Print: PRINT expression
            expr = stmt[1]
            self.get_expression_type(expr)  # Só verifica se a expressão é válida
        
        elif stmt_type == "input":
            # Input: INPUT()
            pass  # Não há verificação específica necessária
        
        elif stmt_type == "if":
            # If: IF (condition) block else_clause
            condition = stmt[1]
            then_block = stmt[2]
            else_block = stmt[3] if len(stmt) > 3 else None
            
            # Verifica a condição
            condition_type = self.get_expression_type(condition)
            if condition_type and condition_type != 'boolean':
                self.add_error("Condição do IF deve ser uma expressão booleana")
            
            # Analisa o bloco then
            self.symbol_table.enter_scope()
            for s in then_block:
                self.analyze_statement(s)
            self.symbol_table.exit_scope()
            
            # Analisa o bloco else (se existir)
            if else_block:
                self.symbol_table.enter_scope()
                for s in else_block:
                    self.analyze_statement(s)
                self.symbol_table.exit_scope()
    
    def analyze(self, ast):
        """Analisa a árvore sintática abstrata completa"""
        if not ast:
            self.add_error("AST vazia - não há código para analisar")
            return False
        
        if ast[0] != "program":
            self.add_error("AST inválida - nó raiz deve ser 'program'")
            return False
        
        statements = ast[1]
        
        # Analisa cada statement
        for stmt in statements:
            self.analyze_statement(stmt)
        
        return len(self.errors) == 0
    
    def print_results(self):
        """Imprime os resultados da análise"""
        print("\n" + "="*60)
        print("📊 RESULTADOS DA ANÁLISE SEMÂNTICA")
        print("="*60)
        
        if self.errors:
            print(f"❌ {len(self.errors)} erro(s) encontrado(s):")
            for error in self.errors:
                print(f"   • {error}")
        else:
            print("✅ Nenhum erro semântico encontrado!")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} aviso(s):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # Mostra a tabela de símbolos
        print(f"\n📋 TABELA DE SÍMBOLOS:")
        symbols = self.symbol_table.get_all_symbols()
        for scope_name, scope_symbols in symbols.items():
            if scope_symbols:
                print(f"   {scope_name.upper()}:")
                for name, symbol in scope_symbols.items():
                    status = "✓" if symbol.initialized else "?"
                    print(f"     {status} {name}: {symbol.data_type}")
            else:
                print(f"   {scope_name.upper()}: (vazio)")

# Regras do parser (mantidas do código original)
precedence = (
    ('left', 'SOMA', 'SUB'),
    ('left', 'MULT', 'DIV'),
)

def p_program(p):
    '''program : statements FIM'''
    p[0] = ("program", p[1])

def p_statement_vardecl(p):
    '''statement : type ID ATRIB expression'''
    p[0] = ('vardecl', p[1], p[2], p[4])

def p_type(p):
    '''type : INT
            | CHAR'''
    p[0] = p[1]

def p_statements_multiple(p):
    '''statements : statements statement'''
    p[0] = p[1] + [p[2]]

def p_statements_single(p):
    '''statements : statement'''
    p[0] = [p[1]]

def p_statement_assignment(p):
    '''statement : ID ATRIB expression'''
    p[0] = ("assign", p[1], p[3])

def p_statement_print(p):
    '''statement : PRINT expression'''
    p[0] = ("print", p[2])

def p_statement_input(p):
    '''statement : INPUT LPAREN RPAREN'''
    p[0] = ("input",)

def p_statement_if(p):
    '''statement : IF LPAREN condition RPAREN block else_clause'''
    p[0] = ("if", p[3], p[5], p[6])

def p_block(p):
    '''block : LBRACE statements RBRACE'''
    p[0] = p[2]

def p_else_clause(p):
    '''else_clause : ELSE block
                   | empty'''
    if len(p) == 3:
        p[0] = p[2]
    else:
        p[0] = None

def p_expression_binop(p):
    '''expression : expression SOMA expression
                  | expression SUB expression
                  | expression MULT expression
                  | expression DIV expression'''
    p[0] = ("binop", p[2], p[1], p[3])

def p_expression_group(p):
    '''expression : LPAREN expression RPAREN'''
    p[0] = p[2]

def p_expression_number(p):
    '''expression : NUM
                  | FLOAT'''
    p[0] = ("number", p[1])

def p_expression_string(p):
    '''expression : STRING'''
    p[0] = ("string", p[1])

def p_expression_char(p):
    '''expression : CHAR_LITERAL'''
    p[0] = ("char", p[1])


def p_expression_id(p):
    '''expression : ID'''
    p[0] = ("id", p[1])

def p_condition(p):
    '''condition : expression relational_operator expression'''
    p[0] = ("condition", p[2], p[1], p[3])

def p_relational_operator(p):
    '''relational_operator : EQ
                           | NEQ
                           | LT
                           | GT
                           | LTE
                           | GTE'''
    p[0] = p[1]

def p_empty(p):
    'empty :'
    p[0] = None

def p_error(p):
    if p:
        print(f"Erro de sintaxe em '{p.value}' na linha {p.lineno}")
    else:
        print("Erro de sintaxe no fim do arquivo")

# Construção do parser
parser = yacc.yacc()

# Função principal para compilar com análise semântica
def compile_with_semantic_analysis(code):
    """Compila código Pokemon com análise semântica completa"""
    print("🔍 INICIANDO COMPILAÇÃO...")
    print("="*60)
    
    # Análise Léxica e Sintática
    print("1️⃣  ANÁLISE LÉXICA E SINTÁTICA")
    try:
        ast = parser.parse(code, lexer=lexer)
        if ast:
            print("✅ Análise sintática concluída com sucesso!")
            print(f"   AST gerada: {ast}")
            generator = IntermediateCodeGenerator()
            generator.generate(ast)
            generator.print_code()

        else:
            print("❌ Falha na análise sintática!")
            return None
    except Exception as e:
        print(f"❌ Erro na análise sintática: {e}")
        return None
    
    # Análise Semântica
    print("\n2️⃣  ANÁLISE SEMÂNTICA")
    analyzer = SemanticAnalyzer()
    success = analyzer.analyze(ast)
    analyzer.print_results()
    
    if success:
        print(f"\n🎉 COMPILAÇÃO CONCLUÍDA COM SUCESSO!")
        return ast
    else:
        print(f"\n💥 COMPILAÇÃO FALHOU!")
        return None

class IntermediateCodeGenerator:
    def __init__(self):
        self.code = []
        self.temp_count = 0
        self.label_count = 0

    def new_temp(self):
        self.temp_count += 1
        return f"t{self.temp_count}"
    
    def new_label(self):
        self.label_count += 1
        return f"L{self.label_count}"

    def generate(self, node):
        if not node:
            return None

        node_type = node[0]

        if node_type == 'program':
            for stmt in node[1]:
                self.generate(stmt)

        elif node_type == 'vardecl':
            var_type, var_name, expr = node[1], node[2], node[3]
            # Declara a variável
            self.code.append(f"declare {var_type} {var_name}")
            # Gera código para a expressão de inicialização
            temp = self.generate(expr)
            self.code.append(f"{var_name} = {temp}")

        elif node_type == 'assign':
            var_name, expr = node[1], node[2]
            temp = self.generate(expr)
            self.code.append(f"{var_name} = {temp}")

        elif node_type == 'print':
            expr = node[1]
            temp = self.generate(expr)
            self.code.append(f"print {temp}")

        elif node_type == 'input':
            temp = self.new_temp()
            self.code.append(f"{temp} = input()")
            return temp

        elif node_type == 'number':
            temp = self.new_temp()
            self.code.append(f"{temp} = {node[1]}")
            return temp

        elif node_type == 'string':
            temp = self.new_temp()
            self.code.append(f"{temp} = \"{node[1]}\"")
            return temp

        elif node_type == 'char':
            temp = self.new_temp()
            self.code.append(f"{temp} = '{node[1]}'")
            return temp

        elif node_type == 'id':
            return node[1]

        elif node_type == 'binop':
            op = node[1]
            left = self.generate(node[2])
            right = self.generate(node[3])
            temp = self.new_temp()
            self.code.append(f"{temp} = {left} {self.translate_op(op)} {right}")
            return temp

        elif node_type == 'condition':
            op = node[1]
            left = self.generate(node[2])
            right = self.generate(node[3])
            temp = self.new_temp()
            self.code.append(f"{temp} = {left} {self.translate_comparison(op)} {right}")
            return temp

        elif node_type == 'if':
            condition = node[1]
            then_block = node[2]
            else_block = node[3] if len(node) > 3 else None
            
            # Gera código para a condição
            condition_temp = self.generate(condition)
            
            # Cria labels
            else_label = self.new_label()
            end_label = self.new_label()
            
            # Pula para else se condição for falsa
            self.code.append(f"if_false {condition_temp} goto {else_label}")
            
            # Código do bloco then
            for stmt in then_block:
                self.generate(stmt)
            
            # Pula para o fim após executar then
            if else_block:
                self.code.append(f"goto {end_label}")
            
            # Label do else
            self.code.append(f"{else_label}:")
            
            # Código do bloco else (se existir)
            if else_block:
                for stmt in else_block:
                    self.generate(stmt)
            
            # Label do fim
            if else_block:
                self.code.append(f"{end_label}:")

    def translate_op(self, op):
        """Traduz operadores aritméticos Pokemon para símbolos padrão"""
        return {
            'PLUSLE': '+',
            'MINUN': '-', 
            'MEWTWO': '*',
            'SCYTHER': '/'
        }.get(op, op)
    
    def translate_comparison(self, op):
        """Traduz operadores de comparação Pokemon para símbolos padrão"""
        return {
            'EEVEE': '==',
            'UMBREON': '!=',
            'FLAREON': '<',
            'JOLTEON': '>',
            'ESPEON': '<=',
            'SYLVIEON': '>='
        }.get(op, op)

    def print_code(self):
        print("\n🔧 CÓDIGO INTERMEDIÁRIO GERADO:")
        print("-" * 40)
        for i, line in enumerate(self.code, 1):
            print(f"{i:2d}: {line}")
        print("-" * 40)


# Testes do compilador
if __name__ == "__main__":
    print("🐾 COMPILADOR POKEMON - ANÁLISE SEMÂNTICA")
    print("="*60)
    
    # Teste 1: Código correto
    print("\n🧪 TESTE 1: Código semanticamente correto")
    codigo_correto = '''
    VAPOREON POKEDEXidade POKEBALL NUMEL20
    VAPOREON POKEDEXresultado POKEBALL POKEDEXidade PLUSLE NUMEL5
    SMEARGLE POKEDEXresultado
    GIRATINA
    '''
    compile_with_semantic_analysis(codigo_correto)
    
    # Teste 2: Variável não declarada
    print("\n\n🧪 TESTE 2: Erro - Variável não declarada")
    codigo_erro1 = '''
    VAPOREON POKEDEXidade POKEBALL NUMEL20
    SMEARGLE POKEDEXnome_inexistente
    GIRATINA
    '''
    compile_with_semantic_analysis(codigo_erro1)
    
    # Teste 3: Tipos incompatíveis
    print("\n\n🧪 TESTE 3: Erro - Tipos incompatíveis")
    codigo_erro2 = '''
    VAPOREON POKEDEXnumero POKEBALL NUMEL10
    CHARMELEON POKEDEXletra POKEBALL CHARMELEON'A'
    POKEDEXnumero POKEBALL POKEDEXletra
    GIRATINA
    '''
    compile_with_semantic_analysis(codigo_erro2)
    
    # Teste 4: Estrutura condicional
    print("\n\n🧪 TESTE 4: Estrutura condicional válida")
    codigo_if = '''
    VAPOREON POKEDEXx POKEBALL NUMEL10
    VAPOREON POKEDEXy POKEBALL NUMEL20
    GREATBALL (POKEDEXx FLAREON POKEDEXy) {
        SMEARGLE POKEDEXx
    } MASTERBALL {
        SMEARGLE POKEDEXy
    }
    GIRATINA
    '''
    compile_with_semantic_analysis(codigo_if)