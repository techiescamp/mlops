# React Performance Optimization

## Optimizing Performance

### Memoization
```jsx
const MemoizedComponent = React.memo(MyComponent);
```

### Lazy Loading
```jsx
const LazyComponent = React.lazy(() => import("./LazyComponent"));
```

### useCallback
```jsx
const memoizedCallback = useCallback(() => {
  doSomething(a, b);
}, [a, b]);
