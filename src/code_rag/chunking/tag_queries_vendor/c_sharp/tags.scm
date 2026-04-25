; Vendored tags.scm for C# — upstream tree-sitter-c-sharp does NOT ship
; one, so we maintain a minimal definition-only set here. Extend as needed.
;
; Convention follows nvim-treesitter / Sourcegraph SCIP:
;   @definition.<kind>  → the whole declaration node
;   @name               → the symbol's name node

(namespace_declaration
  name: (_) @name) @definition.namespace

(file_scoped_namespace_declaration
  name: (_) @name) @definition.namespace

(class_declaration
  name: (identifier) @name) @definition.class

(struct_declaration
  name: (identifier) @name) @definition.struct

(record_declaration
  name: (identifier) @name) @definition.class

; Note: `record_struct_declaration` (record struct) exists in newer C#
; grammars only — omit here so the query compiles against older bindings.

(interface_declaration
  name: (identifier) @name) @definition.interface

(enum_declaration
  name: (identifier) @name) @definition.enum

(method_declaration
  name: (identifier) @name) @definition.method

(constructor_declaration
  name: (identifier) @name) @definition.method

(destructor_declaration
  name: (identifier) @name) @definition.method

(operator_declaration
  (_) @name) @definition.method

(property_declaration
  name: (identifier) @name) @definition.method

(delegate_declaration
  name: (identifier) @name) @definition.function
