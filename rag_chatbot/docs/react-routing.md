# React Routing

## React Router
React Router is a library for handling navigation in a React app.

### Installation
```sh
npm install react-router-dom
```

### Example
```jsx
import { BrowserRouter, Route, Switch } from "react-router-dom";

function App() {
  return (
    <BrowserRouter>
      <Switch>
        <Route path="/about" component={About} />
        <Route path="/" component={Home} />
      </Switch>
    </BrowserRouter>
  );
}
```