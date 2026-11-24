"""설치하지 않은 무거운 모듈의 대역. import 는 통과시키되 실제로 쓰면 오류를 낸다(조용히 엉터리 값을 내지 않게).

막는 기능: OASIS 의 임베딩 추천(TwHIN 등)·neo4j, TinyTroupe 의 의미기억 검색(llama_index)·문서/그림 도구.
우리가 쓰는 기능(OASIS 레딧형 최신·추천점수 피드, TinyTroupe 인물의 듣고 말하기)은 이들을 쓰지 않는다.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

# 필요한 것만 막는다: igraph 처럼 '없으면 알아서 빼는' 선택 의존성까지 막으면 오히려 깨진다
HEAVY = ({"llama_index", "matplotlib", "seaborn", "scipy", "ipywidgets", "IPython", "html2text", "pypandoc", "azure", "docx"} if "tiny" in sys.prefix
         else {"torch", "transformers", "sentence_transformers", "sklearn", "neo4j", "unstructured"})


class _Meta(type):
    def __getattr__(cls, n):
        if n.startswith("__"):
            raise AttributeError(n)
        return _blocked(f"{cls.__name__}.{n}")


def _blocked(name):
    """모듈을 읽으며 전역 객체를 만들어 두는 경우가 있어 생성까지는 허용하고, 만들어진 것을 쓰는 순간 오류를 낸다."""
    def fail(self, n, *a):
        if n.startswith("__"):
            raise AttributeError(n)
        raise RuntimeError(f"설치하지 않은 기능을 사용함: {name}.{n}")
    return _Meta(name, (), {"__init__": lambda self, *a, **k: None, "__getattr__": fail, "__call__": lambda self, *a, **k: fail(self, "call"),
                            "__iter__": lambda self: fail(self, "iter"), "__getitem__": lambda self, k: fail(self, "getitem")})


class _Mod(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        if self.__name__ == "torch":
            if n == "no_grad":                       # 모듈을 읽을 때 장식자로 쓰인다
                return lambda: (lambda f: f)
            if n == "cuda":
                return types.SimpleNamespace(is_available=lambda: False)
            if n == "device":
                return str
        return _blocked(f"{self.__name__}.{n}")


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in HEAVY:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        return _Mod(spec.name)

    def exec_module(self, m):
        m.__path__ = []


for _n in list(HEAVY):                                # 실제로 설치돼 있으면 진짜를 쓴다
    if importlib.util.find_spec(_n) is not None:
        HEAVY.discard(_n)
sys.meta_path.append(_Finder())
