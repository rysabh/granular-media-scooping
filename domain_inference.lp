state(state_0000).
state(state_0001).
state(state_0002).
state(state_0003).
state(state_0004).
state(state_0005).
state(state_0006).
state(state_0007).
state(state_0008).
state(state_0009).
pred(tool_away).
holds(state_0000,tool_away).
pred(tool_near_pile).
holds(state_0001,tool_near_pile).
pred(tool_contains_material).
holds(state_0003,tool_contains_material).
pred(tool_in_pile).
holds(state_0003,tool_in_pile).
pred(tool_lifted).
holds(state_0005,tool_lifted).
pred(tool_at_dump).
holds(state_0007,tool_at_dump).
pred(material_deposited).
holds(state_0009,material_deposited).
pred(tool_contains_material).
holds(state_0009,tool_contains_material).
pred(tool_in_pile).
holds(state_0009,tool_in_pile).
action(approach).
transition(state_0000,approach,state_0001).
action(scoop).
transition(state_0001,scoop,state_0003).
action(lift).
transition(state_0003,lift,state_0005).
action(transport).
transition(state_0005,transport,state_0007).
action(dump).
transition(state_0007,dump,state_0009).


% Preconditions: predicates true in source state
pre(A, P) :- transition(S, A, _), holds(S, P).

% Add effects: predicates true in target state but not source state
add(A, P) :- transition(S, A, T), holds(T, P), not holds(S, P).

% Delete effects: predicates true in source state but not target state
del(A, P) :- transition(S, A, T), holds(S, P), not holds(T, P).

#show action/1.
#show pred/1.
#show pre/2.
#show add/2.
#show del/2.
