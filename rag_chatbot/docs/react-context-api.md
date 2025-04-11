# React Context API

## What is Context API?
Context API is a way to manage global state without prop drilling.

### Example
```jsx
const ThemeContext = React.createContext("light");
function App() {
  return (
    <ThemeContext.Provider value="dark">
      <MyComponent />
    </ThemeContext.Provider>
  );
}
```
