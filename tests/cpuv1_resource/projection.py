"""Exact OFF-expression projection; no parenthesis/token normalization."""
import re


def arguments(text, opening):
    depth = 1; start = opening + 1; i = start; args = []
    quote = None
    while i < len(text):
        c = text[i]
        if quote:
            if c == '\\': i += 2; continue
            if c == quote: quote = None
        elif text.startswith('/*', i):
            i = text.index('*/', i + 2) + 2; continue
        elif text.startswith('//', i):
            i = text.find('\n', i + 2)
            if i < 0: raise ValueError('unterminated argument')
            continue
        elif c in ('"', "'"): quote = c
        elif c in '([{': depth += 1
        elif c in ')]}':
            depth -= 1
            if not depth:
                args.append(text[start:i].strip()); return args, i + 1
        elif c == ',' and depth == 1:
            args.append(text[start:i].strip()); start = i + 1
        i += 1
    raise ValueError('unterminated argument')


def off(text):
    text = re.sub(r'/\* CPUV1_RESOURCE_BEGIN \*/.*?/\* CPUV1_RESOURCE_END \*/',
                  '', text, flags=re.S)
    pattern = re.compile(r'\bET_R_(ALLOC|FREE|REALLOC)\s*\(')
    while True:
        m = pattern.search(text)
        if not m: return text
        args, end = arguments(text, m.end() - 1)
        if m[1] == 'FREE':
            assert len(args) == 5
            original = 'free(' + args[2] + ')'
        else:
            assert len(args) == (5 if m[1] == 'ALLOC' else 7)
            original = args[2]
        text = text[:m.start()] + original + text[end:]


def tokens(text):
    # Keep literal contents and every punctuation token, including parentheses.
    pattern = r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|[0-9]+|[^\s]'
    return [m[0] for m in re.finditer(pattern, text, re.S)
            if not m[0].startswith(('/*', '//'))]
