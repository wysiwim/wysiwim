import javalang
import traceback
from PIL import Image
from graphviz import Digraph
import io

# This algorithm was not used in the paper, it allows to generate a full trivial AST visualization

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

attr_blacklist=["_position", "documentation"]
def viz_rec(dot, node, parent = None):
    dot.lid += 1
    ni = "n" + str(dot.lid)
    if isinstance(node, javalang.ast.Node):
        label = str(node)
        if parent:
            dot.edge(parent, ni)
        for attr in vars(node):
            if attr in attr_blacklist:
                continue
            attr_val = getattr(node, attr)
            if attr_val:
                dot.aid += 1
                viz_rec(dot, attr_val, ni)
        dot.node(ni, label)
    elif type(node) == list:
        for item in node:
            viz_rec(dot, item, parent)
    else:
        dot.node(ni, str(node))
        if parent:
            dot.edge(parent, ni)


def from_to_file(in_path, out_path):
    code = load_code(in_path)
    image = generate_ast(code)
    image.save(out_path)


def generate_ast(code):
    ast = parse_program(code)
    dot = Digraph(comment="AST", format='png')
    dot.lid = 0
    dot.aid = 0
    viz_rec(dot, ast)
    image_raw = dot.pipe()
    return Image.open(io.BytesIO(image_raw))


if __name__ == "__main__":
    from_to_file("<path>/example.java", "<path>/example_full_ast.png")
