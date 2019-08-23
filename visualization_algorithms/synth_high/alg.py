import javalang
import traceback
from graphviz import Digraph
from prettify_js import get_js_library
import time
from PIL import Image
import io
import imgkit


def load_code(path):
    code = []
    with open(path, "r") as fin:
        for l in fin:
            code.append(l)
    return "".join(code)


def from_to_file(in_path, out_path, lang):
    code = load_code(in_path)
    image = render(code, lang)
    image.save(out_path)


def render(code, lang):
    # uses https://github.com/google/code-prettify for syntax highlighting
    jslib = get_js_library()
    html_page = """
        <!DOCTYPE html><html>
        <script>"""+ jslib + """</script>
        <!-- custom version of sunburst style, originally by David Leibovic -->
        <style type="text/css">
            pre .str, code .str { color: #65B042; } /* string  - green */
            pre .kwd, code .kwd { color: #E28964; } /* keyword - dark pink */
            pre .com, code .com { color: #AEAEAE; font-style: italic; } /* comment - gray */
            pre .typ, code .typ { color: #89bdff; } /* type - light blue */
            pre .lit, code .lit { color: #3387CC; } /* literal - blue */
            pre .pun, code .pun { color: #fff; } /* punctuation - white */
            pre .pln, code .pln { color: #fff; } /* plaintext - white */
            pre .tag, code .tag { color: #89bdff; } /* html/xml tag    - light blue */
            pre .atn, code .atn { color: #bdb76b; } /* html/xml attribute name  - khaki */
            pre .atv, code .atv { color: #65B042; } /* html/xml attribute value - green */
            pre .dec, code .dec { color: #3387CC; } /* decimal - blue */
    
            body {
                margin: 0px;
            }
            pre {
                margin: 0px;
            }
            
            pre.prettyprint, code.prettyprint {
                background-color: #000;
                padding: none;
                border: none;
                margin: none;
            }
    
            /* Specify class=linenums on a pre to get line numbering */
            ol.linenums { margin-top: 0; margin-bottom: 0; color: #AEAEAE; } /* IE indents via margin-left */
            li.L0,li.L1,li.L2,li.L3,li.L5,li.L6,li.L7,li.L8 { list-style-type: none }
            /* Alternate shading for lines */
            li.L1,li.L3,li.L5,li.L7,li.L9 { }
    
            @media print {
              pre .str, code .str { color: #060; }
              pre .kwd, code .kwd { color: #006; font-weight: bold; }
              pre .com, code .com { color: #600; font-style: italic; }
              pre .typ, code .typ { color: #404; font-weight: bold; }
              pre .lit, code .lit { color: #044; }
              pre .pun, code .pun { color: #440; }
              pre .pln, code .pln { color: #000; }
              pre .tag, code .tag { color: #006; font-weight: bold; }
              pre .atn, code .atn { color: #404; }
              pre .atv, code .atv { color: #060; }
            }
        </style>
        <?prettify lang=%s linenums=false?>
        <pre class="prettyprint lang-java mycode">""" % lang + code.replace("\t", "    ") + """    </pre>
    </html>"""
    options = {
        'format': 'png',
        'quiet': '',
	'width': 50,
    }
    image_raw = imgkit.from_string(html_page, False, options)
    return Image.open(io.BytesIO(image_raw))

if __name__ == "__main__":
    from_to_file("<path>/test.java", "<path>/test.png", "C")
