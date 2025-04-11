# React Forms

## Controlled vs Uncontrolled Components

### Controlled Component
```jsx
function Form() {
  const [input, setInput] = useState("");
  return <input value={input} onChange={(e) => setInput(e.target.value)} />;
}
```

### Uncontrolled Component
```jsx
function Form() {
  const inputRef = useRef();
  return <input ref={inputRef} />;
}
```