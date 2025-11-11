-assumenosideeffects public class androidx.compose.runtime.ComposerKt {
    void sourceInformation(androidx.compose.runtime.Composer,java.lang.String);
    void sourceInformationMarkerStart(androidx.compose.runtime.Composer,int,java.lang.String);
    void sourceInformationMarkerEnd(androidx.compose.runtime.Composer);
}

# Composer's class initializer doesn't do anything but create an EMPTY object. Marking the
# initializers as having no side effects can help encourage shrinkers to merge/devirtualize Composer
# with ComposerImpl.
-assumenosideeffects public class androidx.compose.runtime.Composer {
    void <clinit>();
}
-assumenosideeffects public class androidx.compose.runtime.ComposerImpl {
    void <clinit>();
}