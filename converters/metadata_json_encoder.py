import json


class MetadataJSONEncoder(json.JSONEncoder):
    DEFAULT_INDENT: int = 2

    CONTAINER_TYPES: tuple[type] = (list, tuple, dict)

    ARRAY_ALIKE_TYPES: tuple[type] = (list, tuple)

    def __init__(
        self,
        *,
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=True,
        sort_keys=False,
        indent=None,
        separators=None,
        default=None,
    ):
        if indent is None:
            indent = self.DEFAULT_INDENT
        super().__init__(
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
            default=default,
        )
        self.curr_indent_lvl = 0

    def iterencode(self, o, _one_shot=False) -> str:
        return self.encode(o)

    def encode(self, o: object) -> str:
        if isinstance(o, self.ARRAY_ALIKE_TYPES):
            return self._encode_array_alike_object(o)
        if isinstance(o, dict):
            return self._encode_dict(o)
        return json.dumps(
            o,
            skipkeys=self.skipkeys,
            ensure_ascii=self.ensure_ascii,
            check_circular=self.check_circular,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
            indent=self.indent,
            separators=(self.item_separator, self.key_separator),
            default=self.default if hasattr(self, "default") else None,
        )

    def _encode_array_alike_object(self, o: list | tuple | dict) -> str:
        return "[" + ", ".join(self.encode(item) for item in o) + "]"

    def _encode_dict(self, o: object) -> str:
        if not o:
            return "{}"
        self.curr_indent_lvl += 1
        output = [
            f"{self._get_curr_indent_str()}{json.dumps(k) if k is not None else 'null'}: {self.encode(v)}"
            for k, v in o.items()
        ]
        self.curr_indent_lvl -= 1
        return "{\n" + ",\n".join(output) + "\n" + self._get_curr_indent_str() + "}"

    def _get_curr_indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.curr_indent_lvl * self.indent)
        if isinstance(self.indent, str):
            return self.curr_indent_lvl * self.indent
        raise TypeError(f"indent must be of type: int | str; is of type: {type(self.indent)}")
