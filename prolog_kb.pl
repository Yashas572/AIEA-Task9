% Time requirements
needs_time(cs101, monday10am).
needs_time(cs102, monday10am).
needs_time(math201, tuesday1pm).
needs_time(phy101, monday10am).

% Equipment requirements
needs_equipment(cs101, projector).
needs_equipment(cs102, whiteboard).
needs_equipment(math201, projector).
needs_equipment(phy101, lab).

% Room time and equipment availability
has(roomA, monday10am).
has(roomB, monday10am).
has(roomC, monday10am).
has(roomA, tuesday1pm).
has(roomC, tuesday1pm).
has(roomA, projector).
has(roomB, whiteboard).
has(roomC, lab).
has(roomC, projector).

% Scheduling rule
can_schedule(Class, Room) :-
    needs_time(Class, Time),
    needs_equipment(Class, Equipment),
    has(Room, Time),
    has(Room, Equipment).

