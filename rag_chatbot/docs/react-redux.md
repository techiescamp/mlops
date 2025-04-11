# React Redux

## What is Redux?
Redux is a state management library for React applications.

### Installation
```sh
npm install redux react-redux
```

### Example
```jsx
const reducer = (state = { count: 0 }, action) => {
  switch (action.type) {
    case "INCREMENT":
      return { count: state.count + 1 };
    default:
      return state;
  }
};
```
