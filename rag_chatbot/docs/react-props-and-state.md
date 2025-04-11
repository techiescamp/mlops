# React Props and State

## Props
Props (short for properties) allow data to be passed from one component to another.
```jsx
function Greeting(props) {
  return <h1>Hello, {props.name}!</h1>;
}
```

## State
State is used to manage component-specific data.
```jsx
class Counter extends React.Component {
  constructor(props) {
    super(props);
    this.state = { count: 0 };
  }
  render() {
    return <h1>Count: {this.state.count}</h1>;
  }
}
```
