import typing as ty

type T_Factory[I: ty.Any] = ty.Callable[..., I] | type[I]
type TDecor[T_Factory] = ty.Callable[[T_Factory], T_Factory]
