; Vendored tags.scm for JavaScript. The upstream file uses #not-eq? and
; #select-adjacent! predicates that some tree-sitter Python bindings don't
; honor; we keep our version minimal & predicate-free for cross-version
; reliability.

(function_declaration
  name: (identifier) @name) @definition.function

(generator_function_declaration
  name: (identifier) @name) @definition.function

(class_declaration
  name: (identifier) @name) @definition.class

(method_definition
  name: (property_identifier) @name) @definition.method
