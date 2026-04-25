; Vendored tags.scm for TypeScript — extends the upstream tree-sitter-
; typescript file (which only covers abstract classes and signatures) to
; include concrete classes, function declarations, and method definitions
; that real TypeScript code actually uses.
;
; Captures named per nvim-treesitter / SCIP convention:
;   @definition.<kind>  → the whole declaration node
;   @name               → the symbol's name node
; Reference captures from upstream are intentionally dropped here — the
; chunker only cares about definitions; references are handled by the
; graph extractor.

; --- Functions ---------------------------------------------------------------
(function_declaration
  name: (identifier) @name) @definition.function

(generator_function_declaration
  name: (identifier) @name) @definition.function

(function_signature
  name: (identifier) @name) @definition.function

; --- Methods (concrete and abstract / signature) ---------------------------
(method_definition
  name: (property_identifier) @name) @definition.method

(method_signature
  name: (property_identifier) @name) @definition.method

(abstract_method_signature
  name: (property_identifier) @name) @definition.method

; --- Classes (concrete + abstract) -----------------------------------------
(class_declaration
  name: (type_identifier) @name) @definition.class

(abstract_class_declaration
  name: (type_identifier) @name) @definition.class

; --- Interfaces, types, enums, modules -------------------------------------
(interface_declaration
  name: (type_identifier) @name) @definition.interface

(enum_declaration
  name: (identifier) @name) @definition.enum

(type_alias_declaration
  name: (type_identifier) @name) @definition.type

(module
  name: [(identifier) (string)] @name) @definition.module

(internal_module
  name: (identifier) @name) @definition.module
