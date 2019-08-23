import javalang
import traceback
from graphviz import Digraph
import time
from PIL import Image
import io

def escape(str):
    return (str
        .replace('\\', '\\\\')
        .replace("{","")
        .replace("}", "")
        #.replace("'", "\\\\'")
        #.replace('"', '\\\\"')
        .replace('>', "\>")
        .replace('<', "\<"))

def parse_program(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree


def load_code(path):
    code = []
    with open(path, "r") as fin:
        for l in fin:
            code.append(l)
    return "".join(code)

def as_text(node):
    if isinstance(node, javalang.ast.Node):
        #print(str(node) + " - " + str(vars(node)))
        label = ""
        if (type(node) == javalang.tree.BinaryOperation):
            label = "%s %s %s" % (as_text(node.operandl), node.operator, as_text(node.operandr))
        elif (type(node) == javalang.tree.Assignment):
            label = "%s = %s" % (as_text(node.expressionl), as_text(node.value))
        elif (type(node) == javalang.tree.VariableDeclarator):
            label = "%s = %s" % (as_text(node.name), as_text(node.initializer))
        elif (type(node) == javalang.tree.LocalVariableDeclaration):
            label = "%s %s" % (as_text(node.type), as_text(node.declarators))
        elif (type(node) == javalang.tree.Cast):
            label = "(%s) %s" % (as_text(node.type), as_text(node.expression))
        elif (type(node) == javalang.tree.EnhancedForControl):
            label = "%s : %s" % (as_text(node.var), as_text(node.iterable))
        elif (type(node) == javalang.tree.ForControl):
            label = "%s ; %s ; %s" % (as_text(node.init), as_text(node.condition), as_text(node.update))
        elif (type(node) == javalang.tree.This):
            label = "this"
            for s in node.selectors:
                label += ".%s" % (as_text(s))
        elif (type(node) == javalang.tree.MethodInvocation):
            label = "%s%s%s( %s )" % (
                as_text(node.qualifier),
                "." if node.qualifier else "",
                as_text(node.member),
                as_text(node.arguments)
            )
        else:
            text = ""
            for attr in vars(node):
                if attr in attr_blacklist:
                    continue
                attr_val = getattr(node, attr)
                if attr_val:
                    text += (" " if text else "") + as_text(attr_val)
            label = text
        if "prefix_operators" in vars(node) and node.prefix_operators:
            preops = "".join([as_text(preo) for preo in node.prefix_operators])
            label = preops + label
        if "postfix_operators" in vars(node) and node.postfix_operators:
            posops = "".join([as_text(poso) for poso in node.postfix_operators])
            label = label + posops
        return label

    elif type(node) is list:
        text = ""
        for item in node:
            text += (", " if text else "") + as_text(item)
        return text
    elif node is None:
        return ""
    return str(node)

attr_blacklist=["_position", "documentation"]

def viz_rec(dot, node, parent = None):
    dot.lid += 1
    ni = "n" + str(dot.lid)
    if node is None:
        return
    elif isinstance(node, javalang.ast.Node):
        if parent and not (type(node) == javalang.tree.BlockStatement) :
            dot.edge(parent, ni)
        if (type(node) is javalang.tree.MethodDeclaration):
            label = "%s %s %s (%s)" % (as_text(node.modifiers), as_text(node.return_type), as_text(node.name), as_text(node.parameters))
            dot.node(ni, escape(label), shape="record")
            p = ni
            for instr in node.body:
                p = viz_rec(dot, instr, p)

        elif (type(node) == javalang.tree.IfStatement):
            label = as_text(node.condition)
            dot.node(ni, escape(label), shape="diamond")
            viz_rec(dot, node.then_statement, ni)
            viz_rec(dot, node.else_statement, ni)

        elif (type(node) == javalang.tree.WhileStatement):
            label = "while " + as_text(node.condition)
            dot.node(ni, escape(label), shape="doubleoctagon")
            r = viz_rec(dot, node.body, ni)
            dot.edge(r, ni)

        elif (type(node) == javalang.tree.ForStatement):
            label = as_text(node.control)
            dot.node(ni, escape(label), shape="tripleoctagon")
            r = viz_rec(dot, node.body, ni)
            dot.edge(r, ni)

        elif (type(node) == javalang.tree.BlockStatement):
            p = parent
            for instr in node.statements:
                p = viz_rec(dot, instr, p)
            return p

        elif (type(node) == javalang.tree.ReturnStatement):
            label = "return "+as_text(node)
            dot.node(ni, escape(label), shape="cds")

        elif (type(node) == javalang.tree.StatementExpression):
            label = as_text(node)
            dot.node(ni, escape(label))

        elif (type(node) == javalang.tree.LocalVariableDeclaration):
            label = as_text(node)
            dot.node(ni, escape(label))

        #elif (type(n) == javalang.tree.TypeDeclaration):
        #elif (type(n) == javalang.tree.Cast):
        #elif (type(n) == javalang.tree.VariableDeclarator):
        #elif (type(n) == javalang.tree.LocalVariableDeclaration):
        #elif (type(n) == javalang.tree.Literal):
        #elif (type(n) == javalang.tree.Assignment):
        #elif (type(n) == javalang.tree.This):
        #elif (type(n) == javalang.tree.MethodInvocation):
        #elif (type(n) == javalang.tree.WhileStatement):
        #elif (type(n) == javalang.tree.ForStatement):
        #elif (type(n) == javalang.tree.BinaryOperation):
        #elif (type(n) == javalang.tree.MemberReference):
        else:
            label = ""
            for attr in vars(node):
                if attr in attr_blacklist:
                    continue
                attr_val = getattr(node, attr)
                if attr_val:
                    dot.aid += 1
                    str_aid = "a%s"%dot.aid
                    viz_rec(dot, attr_val, ni+":"+str_aid)
                    label += "| <%s>"%str_aid
            dot.node(ni, label, shape="record")
    elif type(node) == list:
        for item in node:
            viz_rec(dot, item, parent)
    else:
        dot.node(ni, str(node))
        if parent:
            dot.edge(parent, ni)
    return ni


def from_to_file(in_path, out_path):
    code = load_code(in_path)
    image = generate_viz(code)
    image.save(out_path)


def generate_viz(code, lang):
    ast = parse_program(code)
    dot = Digraph(comment="AST", format='png')
    dot.lid = 0
    dot.aid = 0
    viz_rec(dot, ast)
    image_raw = dot.pipe()
    return Image.open(io.BytesIO(image_raw))


if __name__ == "__main__":
    from_to_file("<path>/example.java", "<path>/example_simple_ast.png")
