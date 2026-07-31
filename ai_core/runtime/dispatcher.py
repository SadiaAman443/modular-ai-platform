"""Event dispatching and subscription service for the Conversation Runtime.

This module provides a publish-subscribe EventDispatcher that allows decoupled
transport layers, loggers, or analytics engines to subscribe to runtime events.
"""

import inspect
from typing import Any, Callable, Coroutine, Optional, Union

from ai_core.runtime.events import Event, EventType

# Type alias for event callbacks that may return None, a single Event, or a list of Events.
EventCallbackResult = Union[None, Event, list[Event]]
SyncEventCallback = Callable[[Event], EventCallbackResult]
AsyncEventCallback = Callable[[Event], Coroutine[Any, Any, EventCallbackResult]]
EventCallback = Union[SyncEventCallback, AsyncEventCallback]


class EventDispatcher:
    """Pub-sub event dispatcher for routing incoming and outgoing runtime events.

    Attributes:
        _type_listeners: Mapping from EventType to lists of callbacks.
        _global_listeners: List of callbacks subscribed to all event types.
    """

    def __init__(self) -> None:
        """Initializes an empty EventDispatcher."""
        self._type_listeners: dict[EventType, list[EventCallback]] = {
            et: [] for et in EventType
        }
        self._global_listeners: list[EventCallback] = []

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """Subscribes a callback to a specific EventType.

        Args:
            event_type: The EventType to listen for.
            callback: Function or coroutine invoked when the event is dispatched.
        """
        if callback not in self._type_listeners[event_type]:
            self._type_listeners[event_type].append(callback)

    def subscribe_all(self, callback: EventCallback) -> None:
        """Subscribes a callback to all dispatched EventTypes.

        Args:
            callback: Function or coroutine invoked on any event dispatch.
        """
        if callback not in self._global_listeners:
            self._global_listeners.append(callback)

    def unsubscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """Removes a callback subscription from a specific EventType.

        Args:
            event_type: The target EventType.
            callback: The callback to remove.
        """
        if callback in self._type_listeners.get(event_type, []):
            self._type_listeners[event_type].remove(callback)

    def unsubscribe_all(self, callback: EventCallback) -> None:
        """Removes a callback from global subscriptions.

        Args:
            callback: The callback to remove.
        """
        if callback in self._global_listeners:
            self._global_listeners.remove(callback)

    def clear(self) -> None:
        """Removes all registered listeners."""
        for et in self._type_listeners:
            self._type_listeners[et].clear()
        self._global_listeners.clear()

    def _collect_listeners(self, event_type: EventType) -> list[EventCallback]:
        """Returns the ordered list of callbacks for the given event type."""
        return list(self._type_listeners.get(event_type, [])) + list(self._global_listeners)

    def _process_result(
        self, result: EventCallbackResult, collected: list[Event]
    ) -> None:
        if isinstance(result, Event):
            collected.append(result)
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, Event):
                    collected.append(item)

    def dispatch(self, event: Event) -> list[Event]:
        """Synchronously dispatches an event to all matching listeners.

        Args:
            event: The Event instance to dispatch.

        Returns:
            A list of secondary Event instances returned by callbacks.

        Raises:
            RuntimeError: If a coroutine callback is dispatched synchronously.
        """
        listeners = self._collect_listeners(event.event_type)
        collected_events: list[Event] = []

        for callback in listeners:
            if inspect.iscoroutinefunction(callback):
                raise RuntimeError(
                    f"Cannot synchronously dispatch to async callback {callback.__name__}. "
                    "Use adispatch() instead."
                )
            result = callback(event)
            self._process_result(result, collected_events)

        return collected_events

    async def adispatch(self, event: Event) -> list[Event]:
        """Asynchronously dispatches an event to all matching listeners.

        Supports both synchronous functions and asynchronous coroutine callbacks.

        Args:
            event: The Event instance to dispatch.

        Returns:
            A list of secondary Event instances returned by callbacks.
        """
        listeners = self._collect_listeners(event.event_type)
        collected_events: list[Event] = []

        for callback in listeners:
            if inspect.iscoroutinefunction(callback):
                result = await callback(event)
            else:
                result = callback(event)
            self._process_result(result, collected_events)

        return collected_events
