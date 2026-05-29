(define (domain scooping)
  (:predicates
    (material_deposited)
    (tool_at_dump)
    (tool_away)
    (tool_contains_material)
    (tool_in_pile)
    (tool_lifted)
    (tool_near_pile)
  )

  (:action approach
    :precondition (and
      (tool_away)
    )
    :effect (and
      (not (tool_away))
      (tool_near_pile)
    )
  )

  (:action dump
    :precondition (and
      (tool_at_dump)
    )
    :effect (and
      (not (tool_at_dump))
      (material_deposited)
      (tool_contains_material)
      (tool_in_pile)
    )
  )

  (:action lift
    :precondition (and
      (tool_contains_material)
      (tool_in_pile)
    )
    :effect (and
      (not (tool_contains_material))
      (not (tool_in_pile))
      (tool_lifted)
    )
  )

  (:action scoop
    :precondition (and
      (tool_near_pile)
    )
    :effect (and
      (not (tool_near_pile))
      (tool_contains_material)
      (tool_in_pile)
    )
  )

  (:action transport
    :precondition (and
      (tool_lifted)
    )
    :effect (and
      (not (tool_lifted))
      (tool_at_dump)
    )
  )

)
