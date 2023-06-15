import { IMAGE_CHANGE } from './actions';

const initialState = {
    image: "",
};

const imageReducer = (state = initialState, action) => {
    switch (action.type) {
        case IMAGE_CHANGE:
            return {
                ...state,
                image: action.image
            }
        default:
            return state;
    }
}

export default imageReducer;